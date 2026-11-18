// SE Matte – Fast mode helper (macOS).
//
// Hardware decode (AVFoundation) → Robust Video Matting on Core ML (true soft alpha + spill-free edge
// colour, stable over time; Apple Vision person segmentation as fallback) → look controls (Core Image)
// → ProRes 4444 with straight alpha via the hardware encoder. All local, 50–90 fps at 1080p on an M4.
//
//   semfast render  --src V --out O.mov --start N --end M [--model rvm.mlmodelc --units gpu|ane|all]
//                   [--quality balanced|accurate] [--choke px] [--feather px] [--contrast 0..1]
//   semfast preview --src V --frame N --out mask.png [--max 960] (+ model / look options)
//   semfast frame   --src V --frame N --out frame.jpg [--max 960]
//   semfast serve   one JSON request per stdin line → one JSON reply per stdout line; keeps models and
//                   videos open so panel previews take tens of ms instead of a process start + model load.
//                   {"cmd":"preview"|"frame", "src":…, "frame":N, "max":960, "model":…, "units":…,
//                    "quality":…, "choke":…, "feather":…, "contrast":…}  →  {"ok":true,"data":"<base64>",…}
//
// render prints "P <done> <total>" lines, then "DONE <frames>"; errors go to stderr ("ERR ...").
// Videos stored sideways with a rotation flag (phone portrait clips) are processed upright, the way
// After Effects and ffmpeg display them.

import AVFoundation
import AppKit
import CoreImage
import CoreML
import Accelerate
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers
import Vision
import VideoToolbox

// ---------------------------------------------------------------- options
struct Opts {
    var cmd = ""
    var src = "", out = ""
    var start = 0, end = -1, frame = 0, maxSide = 960
    var quality = "balanced"
    var codec = "prores4444"      // prores4444 | hevc (HEVC with alpha: ~10× smaller, but AE 2024 can't read it)
    var refine = true
    var raw = false               // preview: write Vision's own low-res mask (debug / tuning)
    var model = ""                // compiled RVM Core ML model (.mlmodelc); empty → Apple Vision
    var units = "gpu"             // gpu | ane | all  (where Core ML runs RVM)
    var choke: Float = 0, feather: Float = 0, contrast: Float = 0
    var tighten: Float = 0        // 0..1: squeeze RVM's soft edge band (1 → levels 0.2–0.8: no faint fuzz)
}

func parse() -> Opts {
    var o = Opts()
    var a = Array(CommandLine.arguments.dropFirst())
    guard !a.isEmpty else { fail("usage: semfast render|preview|frame|serve --src ...") }
    o.cmd = a.removeFirst()
    var i = 0
    while i < a.count {
        let k = a[i], v = i + 1 < a.count ? a[i + 1] : ""
        switch k {
        case "--src": o.src = v; i += 2
        case "--out": o.out = v; i += 2
        case "--start": o.start = Int(v) ?? 0; i += 2
        case "--end": o.end = Int(v) ?? -1; i += 2
        case "--frame": o.frame = Int(v) ?? 0; i += 2
        case "--max": o.maxSide = Int(v) ?? 960; i += 2
        case "--quality": o.quality = v; i += 2
        case "--codec": o.codec = v; i += 2
        case "--choke": o.choke = Float(v) ?? 0; i += 2
        case "--feather": o.feather = Float(v) ?? 0; i += 2
        case "--contrast": o.contrast = Float(v) ?? 0; i += 2
        case "--no-refine": o.refine = false; i += 1
        case "--raw": o.raw = true; i += 1
        case "--model": o.model = v; i += 2
        case "--units": o.units = v; i += 2
        case "--tighten": o.tighten = Float(v) ?? 0; i += 2
        default: i += 1
        }
    }
    return o
}

struct Failure: Error { let msg: String }

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write("ERR \(msg)\n".data(using: .utf8)!)
    exit(1)
}

func say(_ s: String) {
    FileHandle.standardOutput.write((s + "\n").data(using: .utf8)!)
}

let ci = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .cacheIntermediates: false,
                             .workingFormat: CIFormat.RGBAf])

// ---------------------------------------------------------------- edge colour (blur-fusion, on the GPU)
// Foreground colour at semi-transparent edges without the background's colour mixed in
// (Forte & Pitié 2021, "Approximate fast foreground colour estimation"): two box-blur passes.
// Keeps the source's own detail (hair strands, skin texture) – unlike RVM's own foreground output,
// which is soft. Same method Pro mode uses (refine.estimate_foreground).
let kPackF = CIColorKernel(source: "kernel vec4 k(__sample f, __sample m) { float a = m.r; return vec4(f.rgb * a, a); }")!
let kPackB = CIColorKernel(source: "kernel vec4 k(__sample b, __sample m) { float a = 1.0 - m.r; return vec4(b.rgb * a, a); }")!
let kFuse = CIColorKernel(source: """
kernel vec4 k(__sample I, __sample bf, __sample bb, __sample m) {
  float a = m.r;
  vec3 F = bf.rgb / max(bf.a, 0.0001);
  vec3 B = bb.rgb / max(bb.a, 0.0001);
  return vec4(clamp(F + a * (I.rgb - a * F - (1.0 - a) * B), 0.0, 1.0), 1.0);
}
""")!
let kUnpack = CIColorKernel(source: "kernel vec4 k(__sample b) { return vec4(b.rgb / max(b.a, 0.0001), 1.0); }")!

func boxBlur(_ img: CIImage, size: Int, _ ext: CGRect) -> CIImage {
    img.clampedToExtent().applyingFilter("CIBoxBlur", parameters: [kCIInputRadiusKey: Float(size) / 2]).cropped(to: ext)
}

/// I: upright frame, m: alpha (red) → foreground colour image (alpha 1)
func blurFusion(_ I: CIImage, _ m: CIImage) -> CIImage {
    let ext = I.extent
    var F = I, B = I
    for r in [90, 6] {
        let bf = boxBlur(kPackF.apply(extent: ext, arguments: [F, m])!, size: r, ext)
        let bb = boxBlur(kPackB.apply(extent: ext, arguments: [B, m])!, size: r, ext)
        F = kFuse.apply(extent: ext, arguments: [I, bf, bb, m])!
        B = kUnpack.apply(extent: ext, arguments: [bb])!
    }
    return F
}

func makePool(_ w: Int, _ h: Int, _ fmt: OSType) -> CVPixelBufferPool {
    var pool: CVPixelBufferPool? = nil
    CVPixelBufferPoolCreate(nil, nil, [
        kCVPixelBufferPixelFormatTypeKey as String: fmt,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &pool)
    return pool!
}

func newBuffer(_ pool: CVPixelBufferPool) -> CVPixelBuffer {
    var b: CVPixelBuffer? = nil
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &b)
    return b!
}

// ---------------------------------------------------------------- video source
/// One video file: frame timing, rotation, and upright BGRA frames.
final class Source {
    let asset: AVURLAsset
    let track: AVAssetTrack
    let fd: CMTime            // exact frame duration
    let first: CMTime         // PTS of the first displayed frame
    let W: Int, H: Int        // displayed (upright) size
    let orient: CGImagePropertyOrientation?   // nil: already upright
    private var pool: CVPixelBufferPool? = nil

    init(_ path: String) throws {
        guard FileManager.default.fileExists(atPath: path) else { throw Failure(msg: "File not found: \(path)") }
        asset = AVURLAsset(url: URL(fileURLWithPath: path))
        guard let t = asset.tracks(withMediaType: .video).first else { throw Failure(msg: "No video track.") }
        track = t
        fd = frameDuration(t)
        first = firstPTS(asset, t)
        let cw = Int(t.naturalSize.width), ch = Int(t.naturalSize.height)
        let m = t.preferredTransform
        let ang = (Int((atan2(Double(m.b), Double(m.a)) * 180 / .pi).rounded()) + 360) % 360
        switch ang {
        case 90: orient = .right
        case 180: orient = .down
        case 270: orient = .left
        default: orient = nil
        }
        (W, H) = (ang == 90 || ang == 270) ? (ch, cw) : (cw, ch)
    }

    /// time half a frame into frame n (AVAssetReader starts on the frame overlapping its range start,
    /// so this lands on frame n even when the container's timestamps are slightly rounded)
    func time(of n: Int) -> CMTime {
        CMTimeAdd(CMTimeAdd(first, CMTimeMultiply(fd, multiplier: Int32(n))), CMTimeMultiplyByRatio(fd, multiplier: 1, divisor: 2))
    }

    func readerOutput() -> AVAssetReaderTrackOutput {
        let o = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]])
        o.alwaysCopiesSampleData = false
        return o
    }

    /// decoded buffer → upright BGRA (same buffer when no rotation is needed)
    func upright(_ pb: CVPixelBuffer) -> CVPixelBuffer {
        guard let o = orient else { return pb }
        if pool == nil { pool = makePool(W, H, kCVPixelFormatType_32BGRA) }
        var img = CIImage(cvPixelBuffer: pb).oriented(o)
        img = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
        let out = newBuffer(pool!)
        ci.render(img, to: out)
        return out
    }

    /// one upright frame
    func read(_ n: Int) throws -> CVPixelBuffer {
        let t = time(of: n)
        let r = try AVAssetReader(asset: asset)
        let ro = readerOutput()
        r.timeRange = CMTimeRange(start: t, duration: CMTimeMultiply(fd, multiplier: 2))
        r.add(ro)
        guard r.startReading() else { throw Failure(msg: "Can't decode video: \(String(describing: r.error))") }
        defer { r.cancelReading() }
        let earliest = CMTimeSubtract(t, fd)
        while let sb = ro.copyNextSampleBuffer() {
            if CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sb), earliest) >= 0, let pb = CMSampleBufferGetImageBuffer(sb) {
                return upright(pb)
            }
        }
        throw Failure(msg: "Can't read frame \(n).")
    }
}

/// presentation time of the first displayed frame. Many MP4s carry an edit-list offset (e.g. 2 frames),
/// while After Effects and ffmpeg count frames from the first displayed one, so we align to it.
func firstPTS(_ asset: AVAsset, _ track: AVAssetTrack) -> CMTime {
    guard let r = try? AVAssetReader(asset: asset) else { return .zero }
    let o = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    r.add(o)
    guard r.startReading(), let sb = o.copyNextSampleBuffer() else { return .zero }
    let t = CMSampleBufferGetPresentationTimeStamp(sb)
    r.cancelReading()
    return t.isValid ? t : .zero
}

/// exact frame duration. Containers often store a rounded one (e.g. 33333 µs for 30 fps), which drifts
/// over thousands of frames, so snap to the standard rates when the nominal rate matches one.
func frameDuration(_ track: AVAssetTrack) -> CMTime {
    let fps = Double(track.nominalFrameRate)
    let standard: [(Double, CMTime)] = [
        (23.976, CMTime(value: 1001, timescale: 24000)), (24, CMTime(value: 1, timescale: 24)),
        (25, CMTime(value: 1, timescale: 25)), (29.97, CMTime(value: 1001, timescale: 30000)),
        (30, CMTime(value: 1, timescale: 30)), (48, CMTime(value: 1, timescale: 48)),
        (50, CMTime(value: 1, timescale: 50)), (59.94, CMTime(value: 1001, timescale: 60000)),
        (60, CMTime(value: 1, timescale: 60)), (120, CMTime(value: 1, timescale: 120))]
    for (r, d) in standard where abs(fps - r) < 0.01 { return d }
    let d = track.minFrameDuration
    if d.isValid && d.value > 0 { return d }
    return CMTime(value: 1000, timescale: CMTimeScale((fps > 0 ? fps : 25) * 1000))
}

// ---------------------------------------------------------------- RVM (Robust Video Matting, Core ML)
/// Recurrent video matting: each step takes an upright BGRA frame and returns fp16 tensors
/// (foreground colour with background spill removed 1×3×H×W, alpha 1×1×H×W). State carries over.
final class RVM {
    let model: MLModel
    let W: Int, H: Int
    var state: [String: MLMultiArray] = [:]
    let src: MLMultiArray
    let src2: MLMultiArray          // second input slot: the next frame is prepared while this one runs
    var planes: [UnsafeMutableRawPointer]   // B, G, R, A planar8 scratch
    /// low-res model (format 3): takes the already-downsampled frame and returns the guided filter's
    /// A, b for alpha at low res; the full-res step (alpha = A·gray + b) runs here on the CPU, in
    /// parallel with the next prediction. ~1.6× faster on the Neural Engine than doing it in the model.
    let low: Bool
    let sw: Int, sh: Int
    private var dx: [(Int, Int, Float)] = [], dy: [(Int, Int, Float)] = []   // full → small (bilinear taps)
    private var ux: [(Int, Int, Float)] = [], uy: [(Int, Int, Float)] = []   // small → full

    /// PyTorch bilinear (align_corners=False) source taps for `outN` samples over `inN` with `scale` = in/out
    static func taps(_ outN: Int, _ inN: Int, _ scale: Double) -> [(Int, Int, Float)] {
        (0..<outN).map { i in
            let s = max((Double(i) + 0.5) * scale - 0.5, 0)
            let i0 = min(Int(s), inN - 1), i1 = i0 < inN - 1 ? i0 + 1 : i0
            return (i0, i1, Float(s - Double(i0)))
        }
    }

    init(path: String, units: String, width: Int, height: Int) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units == "ane" ? .cpuAndNeuralEngine : (units == "all" ? .all : .cpuAndGPU)
        do { model = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg) }
        catch { throw Failure(msg: "Can't load the matting model: \(error)") }
        W = width; H = height
        let ins = model.modelDescription.inputDescriptionsByName
        if let c = ins["sm"]?.multiArrayConstraint, c.shape.count == 4, let a = try? MLMultiArray(shape: c.shape, dataType: c.dataType) {
            low = true; src = a; src2 = try! MLMultiArray(shape: c.shape, dataType: c.dataType)
            sh = c.shape[2].intValue; sw = c.shape[3].intValue
            let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
            let ds = Double(meta?["ds"] ?? "") ?? Double(sh) / Double(height)
            if Int(Double(height) * ds) != sh || Int(Double(width) * ds) != sw {
                throw Failure(msg: "Model does not match the video size \(W)x\(H).")
            }
            dx = RVM.taps(sw, width, 1 / ds); dy = RVM.taps(sh, height, 1 / ds)
            ux = RVM.taps(width, sw, Double(sw) / Double(width)); uy = RVM.taps(height, sh, Double(sh) / Double(height))
        } else {
            guard let c = ins["src"]?.multiArrayConstraint,
                  c.shape.count == 4, c.shape[2].intValue == H, c.shape[3].intValue == W,
                  let a = try? MLMultiArray(shape: c.shape, dataType: c.dataType) else {
                throw Failure(msg: "Model does not match the video size \(W)x\(H).")
            }
            low = false; src = a; src2 = try! MLMultiArray(shape: c.shape, dataType: c.dataType); sw = W; sh = H
        }
        planes = (0..<4).map { _ in UnsafeMutableRawPointer.allocate(byteCount: width * height, alignment: 64) }
        try reset()
    }

    deinit { planes.forEach { $0.deallocate() } }

    func reset() throws {
        for i in 1...4 {
            let name = "r\(i)i"
            guard let c = model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint,
                  let a = try? MLMultiArray(shape: c.shape, dataType: c.dataType) else { throw Failure(msg: "Bad model input \(name)") }
            let bytes = a.count * (c.dataType == .float16 ? 2 : (c.dataType == .double ? 8 : 4))
            a.withUnsafeMutableBytes { p, _ in memset(p.baseAddress!, 0, bytes) }
            state[name] = a
        }
    }

    /// BGRA pixel buffer → src tensor (R, G, B planes, fp16 0..1) with vImage
    /// BGRA frame → small src tensor with PyTorch's bilinear downsample (what the model was trained with)
    func loadSmall(_ pb: CVPixelBuffer, _ src: MLMultiArray) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let st = src.strides.map { $0.intValue }
        let (dx, dy, sw, sh) = (self.dx, self.dy, self.sw, self.sh)
        let n = sw * sh
        let f = UnsafeMutablePointer<Float>.allocate(capacity: 3 * n)   // R, G, B float planes
        defer { f.deallocate() }
        DispatchQueue.concurrentPerform(iterations: sh) { y in
            let (y0, y1, ly) = dy[y]
            let r0 = base + y0 * bpr, r1 = base + y1 * bpr
            for x in 0..<sw {
                let (x0, x1, lx) = dx[x]
                let a = 4 * x0, b = 4 * x1
                for c in 0..<3 {          // tensor R, G, B ← BGRA offsets 2, 1, 0
                    let o = 2 - c
                    let top = Float(r0[a + o]) * (1 - lx) + Float(r0[b + o]) * lx
                    let bot = Float(r1[a + o]) * (1 - lx) + Float(r1[b + o]) * lx
                    f[c * n + y * sw + x] = (top * (1 - ly) + bot * ly) * (1 / 255)
                }
            }
        }
        src.withUnsafeMutableBytes { raw, _ in      // float → fp16 tensor (vImage: works on Intel too)
            for c in 0..<3 {
                var s = vImage_Buffer(data: f + c * n, height: vImagePixelCount(sh), width: vImagePixelCount(sw), rowBytes: sw * 4)
                var d = vImage_Buffer(data: raw.baseAddress! + c * st[1] * 2, height: vImagePixelCount(sh),
                                      width: vImagePixelCount(sw), rowBytes: st[2] * 2)
                vImageConvert_PlanarFtoPlanar16F(&s, &d, vImage_Flags(kvImageNoFlags))
            }
        }
    }

    func load(_ pb: CVPixelBuffer, slot: Int = 0) {
        let src = slot == 0 ? self.src : src2
        if low { loadSmall(pb, src); return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        var inBuf = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(H), width: vImagePixelCount(W),
                                  rowBytes: CVPixelBufferGetBytesPerRow(pb))
        var p8 = planes.map { vImage_Buffer(data: $0, height: vImagePixelCount(H), width: vImagePixelCount(W), rowBytes: W) }
        vImageConvert_ARGB8888toPlanar8(&inBuf, &p8[0], &p8[1], &p8[2], &p8[3], vImage_Flags(kvImageNoFlags))  // B G R A
        let st = src.strides.map { $0.intValue }
        src.withUnsafeMutableBytes { raw, _ in
            for (c, plane) in [(0, 2), (1, 1), (2, 0)] {   // tensor channel ← planar (R←2, G←1, B←0)
                var dst = vImage_Buffer(data: raw.baseAddress! + c * st[1] * 2, height: vImagePixelCount(H),
                                        width: vImagePixelCount(W), rowBytes: st[2] * 2)
                vImageConvert_Planar8toPlanar16F(&p8[plane], &dst, vImage_Flags(kvImageNoFlags))
            }
        }
    }

    /// one recurrent step; returns the raw output (alpha, or A,b for a low-res model) → pass it to alpha()
    func step(_ pb: CVPixelBuffer) throws -> MLMultiArray {
        load(pb)
        return try predict(slot: 0)
    }

    /// run the model on an input slot filled by load(_:slot:)
    func predict(slot: Int) throws -> MLMultiArray {
        var d: [String: MLFeatureValue] = [low ? "sm" : "src": MLFeatureValue(multiArray: slot == 0 ? src : src2)]
        for (k, v) in state { d[k] = MLFeatureValue(multiArray: v) }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: d))
        for i in 1...4 { state["r\(i)i"] = out.featureValue(for: "r\(i)o")!.multiArrayValue! }
        guard let r = out.featureValue(for: low ? "ab" : "pha")?.multiArrayValue else { throw Failure(msg: "Matting returned nothing.") }
        return r
    }

    /// step() output → 8-bit alpha in `mb` (full size). Low-res models: alpha = A·mean(RGB) + b with
    /// A, b upsampled bilinearly (RVM's deep guided filter, done at full res here instead of in the model).
    func alpha(_ out: MLMultiArray, frame pb: CVPixelBuffer, into mb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(mb, [])
        defer { CVPixelBufferUnlockBaseAddress(mb, []) }
        let mp = CVPixelBufferGetBaseAddress(mb)!.assumingMemoryBound(to: UInt8.self)
        let mbpr = CVPixelBufferGetBytesPerRow(mb)
        if !low { planeTo8(out, channel: 0, W: W, H: H, dst: mp, dstRow: mbpr); return }
        let n = sw * sh
        var ab = [Float](repeating: 0, count: 2 * n)
        let st = out.strides.map { $0.intValue }
        let (sw, sh) = (self.sw, self.sh)
        out.withUnsafeBytes { raw in
            ab.withUnsafeMutableBufferPointer { d in
                for c in 0..<2 {
                    var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress! + c * st[1] * 2),
                                          height: vImagePixelCount(sh), width: vImagePixelCount(sw), rowBytes: st[2] * 2)
                    var t = vImage_Buffer(data: d.baseAddress! + c * n, height: vImagePixelCount(sh),
                                          width: vImagePixelCount(sw), rowBytes: sw * 4)
                    vImageConvert_Planar16FtoPlanarF(&s, &t, vImage_Flags(kvImageNoFlags))
                }
            }
        }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let sp = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let sbpr = CVPixelBufferGetBytesPerRow(pb)
        let (ux, uy, W) = (self.ux, self.uy, self.W)
        ab.withUnsafeBufferPointer { abp in
            let A = abp.baseAddress!, B = A + n
            DispatchQueue.concurrentPerform(iterations: H) { y in
                let (y0, y1, ly) = uy[y]
                var ra = [Float](repeating: 0, count: sw), rb = [Float](repeating: 0, count: sw)   // vertical blend first
                for x in 0..<sw {
                    ra[x] = A[y0 * sw + x] * (1 - ly) + A[y1 * sw + x] * ly
                    rb[x] = B[y0 * sw + x] * (1 - ly) + B[y1 * sw + x] * ly
                }
                let s = sp + y * sbpr, d = mp + y * mbpr
                for x in 0..<W {
                    let (x0, x1, lx) = ux[x]
                    let a = ra[x0] * (1 - lx) + ra[x1] * lx
                    let b = rb[x0] * (1 - lx) + rb[x1] * lx
                    let g = (Float(s[4 * x]) + Float(s[4 * x + 1]) + Float(s[4 * x + 2])) * (1 / 765)
                    let v = min(max(a * g + b, 0), 1)
                    d[x] = UInt8(v * 255 + 0.5)
                }
            }
        }
    }
}

/// fp16 tensor plane → 8-bit (0..1 → 0..255) into dst (rowBytes dstRow)
func planeTo8(_ t: MLMultiArray, channel: Int, W: Int, H: Int, dst: UnsafeMutableRawPointer, dstRow: Int) {
    let st = t.strides.map { $0.intValue }
    t.withUnsafeBytes { raw in
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress! + channel * st[1] * 2),
                              height: vImagePixelCount(H), width: vImagePixelCount(W), rowBytes: st[2] * 2)
        var d = vImage_Buffer(data: dst, height: vImagePixelCount(H), width: vImagePixelCount(W), rowBytes: dstRow)
        vImageConvert_Planar16FtoPlanar8(&s, &d, vImage_Flags(kvImageNoFlags))
    }
}

/// RVM's alpha has a wide soft band at the outline (fuzz around hair, faint bits of background next
/// to the body). A levels squeeze drops the faint fuzz and makes the edge crisp but still natural.
func tightenAlpha(_ buf: CVPixelBuffer, _ t: Float) {
    guard t > 0 else { return }
    let lo = 0.2 * min(t, 1), hi = 1 - 0.2 * min(t, 1)
    let lut: [UInt8] = (0..<256).map { v in UInt8((max(0, min(1, (Float(v) / 255 - lo) / (hi - lo))) * 255).rounded()) }
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    var b = vImage_Buffer(data: CVPixelBufferGetBaseAddress(buf), height: vImagePixelCount(CVPixelBufferGetHeight(buf)),
                          width: vImagePixelCount(CVPixelBufferGetWidth(buf)), rowBytes: CVPixelBufferGetBytesPerRow(buf))
    vImageTableLookUp_Planar8(&b, &b, lut, vImage_Flags(kvImageNoFlags))
}

func hasLook(_ o: Opts) -> Bool { o.choke != 0 || o.feather > 0 || o.contrast > 0 }

// ---------------------------------------------------------------- matte
func personRequest(_ q: String) -> VNGeneratePersonSegmentationRequest {
    let r = VNGeneratePersonSegmentationRequest()
    r.qualityLevel = q == "accurate" ? .accurate : (q == "fast" ? .fast : .balanced)
    r.outputPixelFormat = kCVPixelFormatType_OneComponent8
    return r
}

/// mask (any size, value in red) → full-res alpha as a CIImage, with the look controls applied.
/// `refine` upsamples a low-res (Vision) mask against the frame's edges.
func finishMask(_ mask: CVPixelBuffer, frame: CIImage, o: Opts, refine: Bool) -> CIImage {
    var m = CIImage(cvPixelBuffer: mask)
    let ext = frame.extent
    if m.extent.size != ext.size {
        if refine {
            let f = CIFilter.edgePreserveUpsample()
            f.inputImage = frame
            f.smallImage = m
            f.spatialSigma = 3
            f.lumaSigma = 0.15
            m = f.outputImage!.cropped(to: ext)
        } else {
            m = m.transformed(by: CGAffineTransform(scaleX: ext.width / m.extent.width, y: ext.height / m.extent.height))
        }
    }
    // same look controls as Pro mode (refine.shape)
    let k = Int((abs(o.choke)).rounded())
    if k > 0 {
        // choke > 0 shrinks the matte (erode), < 0 grows it (dilate)
        m = m.clampedToExtent().applyingFilter(o.choke > 0 ? "CIMorphologyMinimum" : "CIMorphologyMaximum",
                                               parameters: [kCIInputRadiusKey: Float(k)]).cropped(to: ext)
    }
    if o.feather > 0 {
        m = m.clampedToExtent().applyingGaussianBlur(sigma: Double(o.feather)).cropped(to: ext)
    }
    if o.contrast > 0 {
        let width = 1.0 - 0.9 * min(o.contrast, 1.0)
        let s = CGFloat(1.0 / width), b = CGFloat(0.5 - 0.5 / width)
        let f = CIFilter.colorMatrix()
        f.inputImage = m
        f.rVector = CIVector(x: s, y: 0, z: 0, w: 0)
        f.biasVector = CIVector(x: b, y: 0, z: 0, w: 0)
        m = f.outputImage!.applyingFilter("CIColorClamp").cropped(to: ext)
    }
    return m
}

func encode(_ img: CIImage, gray: Bool, jpeg: Bool) throws -> Data {
    let ext = img.extent.integral
    let cg = gray ? ci.createCGImage(img, from: ext, format: .L8, colorSpace: CGColorSpaceCreateDeviceGray())
                  : ci.createCGImage(img, from: ext, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
    guard let c = cg else { throw Failure(msg: "Couldn't render the image.") }
    let data = NSMutableData()
    guard let d = CGImageDestinationCreateWithData(data, (jpeg ? UTType.jpeg : UTType.png).identifier as CFString, 1, nil)
    else { throw Failure(msg: "Couldn't encode the image.") }
    CGImageDestinationAddImage(d, c, jpeg ? [kCGImageDestinationLossyCompressionQuality: 0.88] as CFDictionary : nil)
    guard CGImageDestinationFinalize(d) else { throw Failure(msg: "Couldn't encode the image.") }
    return data as Data
}

func scaled(_ img: CIImage, maxSide: Int) -> CIImage {
    let s = min(1.0, CGFloat(maxSide) / max(img.extent.width, img.extent.height))
    return s < 1 ? img.transformed(by: CGAffineTransform(scaleX: s, y: s)) : img
}

// ---------------------------------------------------------------- previews (shared by the CLI and serve)
final class Previewer {
    var sources: [String: Source] = [:]
    var models: [String: RVM] = [:]     // model path → loaded model (kept warm)

    func source(_ path: String) throws -> Source {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let key = path + "|" + String(describing: attrs?[.modificationDate] ?? "") + "|" + String(describing: attrs?[.size] ?? "")
        if let s = sources[key] { return s }
        if sources.count > 4 { sources.removeAll() }
        let s = try Source(path)
        sources[key] = s
        return s
    }

    func rvm(_ path: String, _ units: String, _ W: Int, _ H: Int) throws -> RVM {
        let key = path + "|" + units
        if let m = models[key], m.W == W, m.H == H { return m }
        if models.count >= 2 { models.removeAll() }    // RVM resnet50 is ~100 MB in memory
        let m = try RVM(path: path, units: units, width: W, height: H)
        models[key] = m
        return m
    }

    // last decoded frame and last AI matte: the panel asks for the frame and then its matte, and
    // dragging an Edge-look slider re-asks for the same frame, so both are usually reused
    private var lastFrame: (String, CVPixelBuffer)? = nil
    private var lastMatte: (String, CVPixelBuffer)? = nil

    func frame(_ s: Source, _ path: String, _ n: Int) throws -> CVPixelBuffer {
        let k = "\(path)|\(n)|\(s.W)x\(s.H)"
        if let (key, pb) = lastFrame, key == k { return pb }
        let pb = try s.read(n)
        lastFrame = (k, pb)
        return pb
    }

    /// what Fast mode keeps on this frame: alpha (gray) PNG at ≤ maxSide, plus the share of subject pixels
    func mask(_ o: Opts) throws -> (Data, Double) {
        let s = try source(o.src)
        let pb = try frame(s, o.src, o.frame)
        let frameImg = CIImage(cvPixelBuffer: pb)
        var m: CIImage
        if !o.model.isEmpty {
            let k = "\(o.src)|\(o.frame)|\(o.model)|\(o.units)|\(o.tighten)"
            let ab: CVPixelBuffer
            if let (key, b) = lastMatte, key == k { ab = b } else {
                let r = try rvm(o.model, o.units, s.W, s.H)
                try r.reset()
                var res: MLMultiArray? = nil
                for _ in 0..<4 { res = try r.step(pb) }    // a few steps let the recurrent state settle
                ab = newBuffer(makePoolCached(s.W, s.H))
                r.alpha(res!, frame: pb, into: ab)
                tightenAlpha(ab, o.tighten)
                lastMatte = (k, ab)
            }
            m = finishMask(ab, frame: frameImg, o: o, refine: false)
        } else {
            let req = personRequest(o.quality)
            try VNImageRequestHandler(cvPixelBuffer: pb).perform([req])
            guard let mask = req.results?.first?.pixelBuffer else { throw Failure(msg: "No person found.") }
            m = o.raw ? CIImage(cvPixelBuffer: mask) : finishMask(mask, frame: frameImg, o: o, refine: o.refine)
        }
        m = scaled(m, maxSide: o.maxSide)
        // coverage: mean alpha (CIAreaAverage on a 1×1 result)
        var px = [UInt8](repeating: 0, count: 4)
        let avg = m.applyingFilter("CIAreaAverage", parameters: [kCIInputExtentKey: CIVector(cgRect: m.extent)])
        ci.render(avg, toBitmap: &px, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)
        return (try encode(m, gray: true, jpeg: false), Double(px[0]) / 255)
    }

    func frameJPEG(_ o: Opts) throws -> (Data, Int, Int) {
        let s = try source(o.src)
        let img = scaled(CIImage(cvPixelBuffer: try frame(s, o.src, o.frame)), maxSide: o.maxSide)
        return (try encode(img, gray: false, jpeg: true), Int(img.extent.width.rounded()), Int(img.extent.height.rounded()))
    }

    private var grayPools: [String: CVPixelBufferPool] = [:]
    func makePoolCached(_ w: Int, _ h: Int) -> CVPixelBufferPool {
        let k = "\(w)x\(h)"
        if let p = grayPools[k] { return p }
        let p = makePool(w, h, kCVPixelFormatType_OneComponent8)
        grayPools[k] = p
        return p
    }
}

func runPreview(_ o: Opts) {
    let p = Previewer()
    do {
        let data: Data
        if o.cmd == "frame" { data = try p.frameJPEG(o).0 } else { data = try p.mask(o).0 }
        try data.write(to: URL(fileURLWithPath: o.out))
    } catch let f as Failure { fail(f.msg) } catch { fail("\(error)") }
    say("DONE 1")
}

/// long-running preview server for the engine (one JSON line in → one JSON line out)
func serve() {
    let p = Previewer()
    setvbuf(stdout, nil, _IOLBF, 0)
    func reply(_ d: [String: Any]) {
        let j = (try? JSONSerialization.data(withJSONObject: d)) ?? Data("{\"ok\":false}".utf8)
        FileHandle.standardOutput.write(j + Data("\n".utf8))
    }
    while let line = readLine() {
        if line.isEmpty { continue }
        autoreleasepool {
            guard let d = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else {
                reply(["ok": false, "error": "bad request"]); return
            }
            var o = Opts()
            o.cmd = d["cmd"] as? String ?? "preview"
            o.src = d["src"] as? String ?? ""
            o.frame = (d["frame"] as? NSNumber)?.intValue ?? 0
            o.maxSide = (d["max"] as? NSNumber)?.intValue ?? 960
            o.model = d["model"] as? String ?? ""
            o.units = d["units"] as? String ?? "gpu"
            o.quality = d["quality"] as? String ?? "balanced"
            o.choke = (d["choke"] as? NSNumber)?.floatValue ?? 0
            o.feather = (d["feather"] as? NSNumber)?.floatValue ?? 0
            o.contrast = (d["contrast"] as? NSNumber)?.floatValue ?? 0
            o.tighten = (d["tighten"] as? NSNumber)?.floatValue ?? 0
            let t0 = Date()
            do {
                switch o.cmd {
                case "frame":
                    let (data, w, h) = try p.frameJPEG(o)
                    reply(["ok": true, "data": data.base64EncodedString(), "width": w, "height": h,
                           "ms": Int(Date().timeIntervalSince(t0) * 1000)])
                case "preview":
                    let (data, cov) = try p.mask(o)
                    reply(["ok": true, "data": data.base64EncodedString(), "coverage": cov,
                           "ms": Int(Date().timeIntervalSince(t0) * 1000)])
                case "ping": reply(["ok": true])
                default: reply(["ok": false, "error": "unknown command \(o.cmd)"])
                }
            } catch let f as Failure { reply(["ok": false, "error": f.msg]) } catch { reply(["ok": false, "error": "\(error)"]) }
        }
    }
}

// ---------------------------------------------------------------- render
func render(_ o: Opts) {
    let s: Source
    do { s = try Source(o.src) } catch let f as Failure { fail(f.msg) } catch { fail("\(error)") }
    let W = s.W, H = s.H, fd = s.fd
    let total = o.end - o.start
    if total <= 0 { fail("Empty frame range.") }
    let startT = s.time(of: o.start)
    let earliest = CMTimeSubtract(startT, fd)

    let reader: AVAssetReader
    do { reader = try AVAssetReader(asset: s.asset) } catch { fail("Can't open video: \(error)") }
    let rout = s.readerOutput()
    reader.timeRange = CMTimeRange(start: startT, duration: CMTimeMultiply(fd, multiplier: Int32(total)))
    reader.add(rout)
    guard reader.startReading() else { fail("Can't decode video: \(String(describing: reader.error))") }

    let dst = URL(fileURLWithPath: o.out)
    try? FileManager.default.removeItem(at: dst)
    let writer: AVAssetWriter
    do { writer = try AVAssetWriter(outputURL: dst, fileType: .mov) } catch { fail("Can't write: \(error)") }
    var settings: [String: Any] = [AVVideoCodecKey: AVVideoCodecType.proRes4444, AVVideoWidthKey: W, AVVideoHeightKey: H]
    if o.codec == "hevc" {
        settings[AVVideoCodecKey] = AVVideoCodecType.hevcWithAlpha
        settings[AVVideoCompressionPropertiesKey] = [
            AVVideoQualityKey: 0.9,
            kVTCompressionPropertyKey_TargetQualityForAlpha as String: 0.95,
            kVTCompressionPropertyKey_AlphaChannelMode as String: kVTAlphaChannelMode_StraightAlpha,
        ]
    }
    let win = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    win.expectsMediaDataInRealTime = false
    // frames are written upright already; keep only a non-rotation transform (e.g. a mirror)
    win.transform = s.orient == nil ? s.track.preferredTransform : .identity
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: win, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: W, kCVPixelBufferHeightKey as String: H,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:]])
    // keep the source's exact time base (e.g. 30000/1001) so AE reads 29.97, not 30
    win.mediaTimeScale = fd.timescale
    writer.movieTimeScale = fd.timescale
    writer.add(win)
    guard writer.startWriting() else { fail("Can't start writer: \(String(describing: writer.error))") }
    writer.startSession(atSourceTime: .zero)

    let maskPool = makePool(W, H, kCVPixelFormatType_OneComponent8)
    let req = personRequest(o.quality)
    let seq = VNSequenceRequestHandler()     // sequence handler = temporally stable masks
    let rvm: RVM?
    do { rvm = o.model.isEmpty ? nil : try RVM(path: o.model, units: o.units, width: W, height: H) }
    catch let f as Failure { fail(f.msg) } catch { fail("\(error)") }
    let look = hasLook(o)
    // stage 2 (compose + encode) runs on its own queue while stage 1 mattes the next frame
    let writeQ = DispatchQueue(label: "semfast.write")
    let inflight = DispatchSemaphore(value: 4)
    let group = DispatchGroup()
    var written = 0
    var n = 0
    var lastReport = 0.0
    let colourPool = makePool(W, H, kCVPixelFormatType_32BGRA)   // edge colour (blur-fusion)

    let prof = ProcessInfo.processInfo.environment["SEMFAST_PROFILE"] != nil
    var tDec = 0.0, tPred = 0.0, tWait = 0.0, tCompose = 0.0, tEnc = 0.0
    var tq = Date()
    func nextFrame() -> (CMSampleBuffer, CVPixelBuffer)? {
        while let sb = rout.copyNextSampleBuffer() {
            guard let raw = CMSampleBufferGetImageBuffer(sb) else { continue }
            if CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sb), earliest) < 0 { continue }
            return (sb, s.upright(raw))
        }
        return nil
    }
    let prepQ = DispatchQueue(label: "semfast.prep")
    var slot = 0
    var cur = nextFrame()
    if let r = rvm, let c = cur { r.load(c.1, slot: 0) }
    while n < total, let (sb, pb) = cur {
        tDec += Date().timeIntervalSince(tq)
        // decode and prepare the next frame while this one is in the Neural Engine
        var nxt: (CMSampleBuffer, CVPixelBuffer)? = nil
        let prepG = DispatchGroup()
        if n + 1 < total {
            let other = 1 - slot
            prepQ.async(group: prepG) {
                nxt = nextFrame()
                if let r = rvm, let x = nxt { r.load(x.1, slot: other) }
            }
        }
        var small: CVPixelBuffer? = nil, pha: MLMultiArray? = nil
        if let r = rvm {
            if n == 0 { for _ in 0..<4 { _ = try? r.predict(slot: slot) } }   // settle the recurrent state on frame 1
            let tp = Date()
            do { pha = try r.predict(slot: slot) } catch { fail("Matting failed at frame \(o.start + n): \(error)") }
            tPred += Date().timeIntervalSince(tp)
        } else {
            do { try seq.perform([req], on: pb) } catch { fail("Vision failed at frame \(o.start + n): \(error)") }
            guard let vmask = req.results?.first?.pixelBuffer else { fail("Vision returned no mask.") }
            small = copyBuffer(vmask)        // Vision may reuse its buffer for the next frame
        }
        let outT = CMTimeMultiply(fd, multiplier: Int32(n))
        let (pha1, small1) = (pha, small)
        n += 1
        let tn = Date()
        prepG.wait()
        tDec += Date().timeIntervalSince(tn)
        cur = nxt; slot = 1 - slot
        let tw = Date()
        inflight.wait()
        tWait += Date().timeIntervalSince(tw)
        tq = Date()
        group.enter()
        writeQ.async {
            defer { inflight.signal(); group.leave() }
            let tc = Date()
            // alpha: RVM's own (plus look controls) or Vision's refined; colour at edges: blur-fusion
            var mb = newBuffer(maskPool)
            if let p = pha1 {
                rvm!.alpha(p, frame: pb, into: mb)
                tightenAlpha(mb, o.tighten)
                if look {
                    let shaped = newBuffer(maskPool)
                    ci.render(finishMask(mb, frame: CIImage(cvPixelBuffer: pb), o: o, refine: false), to: shaped)
                    mb = shaped
                }
            } else {
                ci.render(finishMask(small1!, frame: CIImage(cvPixelBuffer: pb), o: o, refine: o.refine), to: mb)
            }
            // spill-free edge colour that keeps the source's detail
            let cb = newBuffer(colourPool)
            ci.render(blurFusion(CIImage(cvPixelBuffer: pb), CIImage(cvPixelBuffer: mb)), to: cb)
            let o2 = newBuffer(adaptor.pixelBufferPool!)
            CVPixelBufferLockBaseAddress(pb, .readOnly); CVPixelBufferLockBaseAddress(o2, []); CVPixelBufferLockBaseAddress(mb, .readOnly)
            CVPixelBufferLockBaseAddress(cb, .readOnly)
            let sp = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            let op = CVPixelBufferGetBaseAddress(o2)!.assumingMemoryBound(to: UInt8.self)
            let mp = CVPixelBufferGetBaseAddress(mb)!.assumingMemoryBound(to: UInt8.self)
            let sbpr = CVPixelBufferGetBytesPerRow(pb)
            let obpr = CVPixelBufferGetBytesPerRow(o2), mbpr = CVPixelBufferGetBytesPerRow(mb)
            let cp = CVPixelBufferGetBaseAddress(cb)!.assumingMemoryBound(to: UInt8.self)
            let cbpr = CVPixelBufferGetBytesPerRow(cb)
            // straight (un-premultiplied) alpha
            DispatchQueue.concurrentPerform(iterations: H) { y in
                let s = sp + y * sbpr, d = op + y * obpr, mm = mp + y * mbpr, c = cp + y * cbpr
                for x in 0..<W {
                    let a = mm[x]
                    if a == 0 {
                        // fully transparent: colour is irrelevant; flat black makes ProRes files smaller
                        d[4 * x] = 0; d[4 * x + 1] = 0; d[4 * x + 2] = 0; d[4 * x + 3] = 0
                    } else if a == 255 {
                        d[4 * x] = s[4 * x]; d[4 * x + 1] = s[4 * x + 1]; d[4 * x + 2] = s[4 * x + 2]; d[4 * x + 3] = 255
                    } else {
                        // edge: foreground colour without background spill (no halo), source detail kept
                        d[4 * x] = c[4 * x]; d[4 * x + 1] = c[4 * x + 1]; d[4 * x + 2] = c[4 * x + 2]; d[4 * x + 3] = a
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, .readOnly); CVPixelBufferUnlockBaseAddress(o2, []); CVPixelBufferUnlockBaseAddress(mb, .readOnly)
            CVPixelBufferUnlockBaseAddress(cb, .readOnly)
            _ = sb  // keep the decoded sample alive until composed
            let ta = Date()
            tCompose += ta.timeIntervalSince(tc)
            while !win.isReadyForMoreMediaData { usleep(300) }
            if !adaptor.append(o2, withPresentationTime: outT) {
                fail("Encoder error: \(String(describing: writer.error))")
            }
            tEnc += Date().timeIntervalSince(ta)
            written += 1
            let now = Date().timeIntervalSince1970
            if now - lastReport > 0.25 || written == total {
                lastReport = now
                say("P \(written) \(total)")
            }
        }
    }
    group.wait()
    if n < total { fail("The video ended early: got \(n) of \(total) frames.") }
    if prof {
        let k = Double(max(n, 1)) / 1000
        FileHandle.standardError.write(String(format: "PROFILE per frame: decode %.1f ms | predict %.1f | waiting for writer %.1f || compose %.1f | encoder wait %.1f\n",
            tDec / k, tPred / k, tWait / k, tCompose / k, tEnc / k).data(using: .utf8)!)
    }
    win.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    if writer.status != .completed { fail("Writing failed: \(String(describing: writer.error))") }
    say("DONE \(written)")
}

func copyBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer {
    let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
    var dst: CVPixelBuffer? = nil
    CVPixelBufferCreate(nil, w, h, CVPixelBufferGetPixelFormatType(src),
                        [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &dst)
    CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(dst!, [])
    let sb = CVPixelBufferGetBytesPerRow(src), db = CVPixelBufferGetBytesPerRow(dst!)
    let sp = CVPixelBufferGetBaseAddress(src)!, dp = CVPixelBufferGetBaseAddress(dst!)!
    for y in 0..<h { memcpy(dp + y * db, sp + y * sb, min(sb, db)) }
    CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst!, [])
    return dst!
}

// ---------------------------------------------------------------- main
signal(SIGTERM) { _ in exit(2) }
signal(SIGPIPE, SIG_IGN)
let opts = parse()
switch opts.cmd {
case "serve": serve()
case "render", "preview", "frame":
    if opts.src.isEmpty || opts.out.isEmpty { fail("--src and --out are required") }
    if opts.cmd == "render" { render(opts) } else { runPreview(opts) }
default: fail("unknown command \(opts.cmd)")
}
