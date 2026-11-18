/* SE Matte – After Effects side (ExtendScript).
   Every function returns a JSON string: {ok:true,...} or {ok:false,error:"..."}. */

function SEM_q(s) {
    s = String(s);
    var out = '"';
    for (var i = 0; i < s.length; i++) {
        var c = s.charAt(i), code = s.charCodeAt(i);
        if (c === '"' || c === '\\') out += '\\' + c;
        else if (c === '\n') out += '\\n';
        else if (c === '\r') out += '\\r';
        else if (c === '\t') out += '\\t';
        else if (code < 32) out += '\\u' + ('0000' + code.toString(16)).slice(-4);
        else out += c;
    }
    return out + '"';
}

function SEM_json(v) {
    if (v === null || v === undefined) return 'null';
    var t = typeof v;
    if (t === 'number') return isFinite(v) ? String(v) : 'null';
    if (t === 'boolean') return v ? 'true' : 'false';
    if (t === 'string') return SEM_q(v);
    if (v instanceof Array) {
        var a = [];
        for (var i = 0; i < v.length; i++) a.push(SEM_json(v[i]));
        return '[' + a.join(',') + ']';
    }
    var parts = [];
    for (var k in v) if (v.hasOwnProperty(k)) parts.push(SEM_q(k) + ':' + SEM_json(v[k]));
    return '{' + parts.join(',') + '}';
}

function SEM_err(msg) { return SEM_json({ ok: false, error: msg }); }

function SEM_ping() { return SEM_json({ ok: true, version: app.version }); }

function SEM_findComp(id) {
    var ai = app.project.activeItem;
    if (ai && ai instanceof CompItem && ai.id === id) return ai;
    for (var i = 1; i <= app.project.numItems; i++) {
        var it = app.project.item(i);
        if (it instanceof CompItem && it.id === id) return it;
    }
    return null;
}

/* SE Matte keeps a marker in the layer comment so a processed layer can be restored / re-processed:
   "[SE Matte] orig=<original footage id>" */
var SEM_MARK = /\[SE Matte\] orig=(\d+)/;

function SEM_origId(L) {
    var m = SEM_MARK.exec(L.comment || '');
    return m ? parseInt(m[1], 10) : (L.source ? L.source.id : -1);
}

function SEM_itemById(id) {
    try { if (app.project.itemByID) return app.project.itemByID(id); } catch (e) {}
    for (var i = 1; i <= app.project.numItems; i++) if (app.project.item(i).id === id) return app.project.item(i);
    return null;
}

/* find the processed layer again even if layers were added/reordered/renamed since */
function SEM_findLayer(comp, ref) {
    var L;
    if (ref.layerIndex >= 1 && ref.layerIndex <= comp.numLayers) {
        L = comp.layer(ref.layerIndex);
        if (L.source && SEM_origId(L) === ref.sourceId) return L;
    }
    for (var i = 1; i <= comp.numLayers; i++) {
        L = comp.layer(i);
        if (L.name === ref.layerName && L.source && SEM_origId(L) === ref.sourceId) return L;
    }
    for (i = 1; i <= comp.numLayers; i++) {
        L = comp.layer(i);
        if (L.source && SEM_origId(L) === ref.sourceId) return L;
    }
    return null;
}

function SEM_srcFrame(L, t, fps) {
    // comp time → source frame of a (non time-remapped) footage layer
    var lt = (t - L.startTime) * 100 / L.stretch;
    return Math.floor(lt * fps + 0.001);
}

function SEM_selectedLayer() {
    try {
        var comp = app.project.activeItem;
        if (!comp || !(comp instanceof CompItem) || comp.selectedLayers.length === 0) {
            // the panel itself has focus → activeItem can be empty; use the comp that has a selected layer
            var found = [];
            for (var i = 1; i <= app.project.numItems; i++) {
                var it = app.project.item(i);
                if (it instanceof CompItem && it.selectedLayers.length > 0) found.push(it);
            }
            if (found.length === 1) comp = found[0];
            else if (found.length > 1) return SEM_err('Layers are selected in several comps. Click the layer once more in its timeline.');
        }
        if (!comp || !(comp instanceof CompItem)) return SEM_err('Open a composition and select a video layer.');
        if (comp.selectedLayers.length !== 1) return SEM_err('Select exactly one video layer in the timeline.');
        var L = comp.selectedLayers[0];
        var processed = SEM_MARK.test(L.comment || '');
        var orig = processed ? SEM_itemById(SEM_origId(L)) : (L.source || null);
        if (processed && !orig) return SEM_err('The original footage of this layer was removed from the project.');
        if (!(L instanceof AVLayer) || !L.source || !(L.source instanceof FootageItem) ||
            !L.source.mainSource || !(L.source.mainSource instanceof FileSource) || !L.source.file)
            return SEM_err('"' + L.name + '" is not a footage layer. Select a video clip (precomps come later).');
        var src = orig;
        if (src.mainSource.isStill) return SEM_err('Still images are not supported yet. Select a video clip.');
        var path = src.file.fsName;
        if (/\.(png|jpe?g|tiff?|exr|dpx|tga)$/i.test(path))
            return SEM_err('Image sequences are not supported yet. Select a video file.');
        if (L.timeRemapEnabled && !processed) return SEM_err('Time remapping is on for this layer. Turn it off (or precompose) for now.');
        if (L.stretch <= 0) return SEM_err('Reversed (negative stretch) layers are not supported yet.');
        var fps = src.frameRate;
        var total = Math.round(src.duration * fps);
        var inF = Math.max(0, SEM_srcFrame(L, L.inPoint, fps));
        var outF = Math.min(total, SEM_srcFrame(L, L.outPoint - 0.5 / comp.frameRate, fps) + 1);
        var cur = SEM_srcFrame(L, comp.time, fps);
        return SEM_json({
            ok: true, compId: comp.id, compName: comp.name, compFps: comp.frameRate,
            layerIndex: L.index, layerName: L.name, sourceId: src.id, path: path,
            width: src.width, height: src.height, fps: fps, frames: total,
            inFrame: inF, outFrame: outF, current: cur, stretch: L.stretch, processed: processed,
            hasEffects: L.property('ADBE Effect Parade').numProperties > 0
        });
    } catch (e) { return SEM_err('AE error: ' + e.toString()); }
}

/* cheap poll: where is the playhead in source frames of the locked layer */
function SEM_layerFrame(refJson) {
    try {
        var ref = eval('(' + refJson + ')');
        var ai = app.project.activeItem;
        if (ai && ai instanceof CompItem && ai.id !== ref.compId) return SEM_json({ ok: true, active: false });
        var comp = SEM_findComp(ref.compId);
        if (!comp) return SEM_json({ ok: true, active: false });
        var L = SEM_findLayer(comp, ref);
        if (!L) return SEM_json({ ok: true, active: false });
        return SEM_json({ ok: true, active: true, frame: SEM_srcFrame(L, comp.time, SEM_itemById(SEM_origId(L)).frameRate), layerIndex: L.index });
    } catch (e) { return SEM_err(e.toString()); }
}

function SEM_gotoFrame(refJson, frame) {
    try {
        var ref = eval('(' + refJson + ')');
        var comp = SEM_findComp(ref.compId);
        if (!comp) return SEM_err('Composition not found.');
        var L = SEM_findLayer(comp, ref);
        if (!L) return SEM_err('Layer not found.');
        var fps = SEM_itemById(SEM_origId(L)).frameRate;
        comp.time = L.startTime + ((frame + 0.5) / fps) * L.stretch / 100;
        try { comp.openInViewer(); } catch (e2) {}
        return SEM_json({ ok: true });
    } catch (e) { return SEM_err(e.toString()); }
}

function SEM_folder() {
    for (var i = 1; i <= app.project.numItems; i++) {
        var it = app.project.item(i);
        if (it instanceof FolderItem && it.name === 'SE Matte') return it;
    }
    return app.project.items.addFolder('SE Matte');
}

/* a render that is already in the project (e.g. "Apply again") is reused instead of imported twice */
function SEM_findImported(path) {
    var target = new File(path).fsName;
    for (var i = 1; i <= app.project.numItems; i++) {
        var it = app.project.item(i);
        if (it instanceof FootageItem && it.file && it.parentFolder && it.parentFolder.name === 'SE Matte' &&
            it.file.fsName === target) return it;
    }
    return null;
}

function SEM_importFile(path, straight) {
    var f = new File(path);
    if (!f.exists) throw new Error('Render not found: ' + path);
    var item = app.project.importFile(new ImportOptions(f));
    item.parentFolder = SEM_folder();
    try { if (straight) item.mainSource.alphaMode = AlphaMode.STRAIGHT; } catch (e) {}
    return item;
}

/* restore a processed layer to its original footage (same layer, nothing else touched) */
function SEM_restoreOn(L) {
    var orig = SEM_itemById(SEM_origId(L));
    if (!orig) throw new Error('Original footage not found in the project.');
    var inP = L.inPoint, outP = L.outPoint, nm = L.name;
    if (L.timeRemapEnabled) L.timeRemapEnabled = false;
    L.replaceSource(orig, false);
    if (L.name !== nm) L.name = nm;
    L.outPoint = outP; L.inPoint = inP;
    if (Math.abs(L.outPoint - outP) > 0.001) L.outPoint = outP;
    L.comment = (L.comment || '').replace(SEM_MARK, '').replace(/\s+$/, '');
}

/* swap L's footage for the cutout `item` (covering original source frames [start,end)), frame-exact.
   The layer keeps its transform, masks, effects, parenting and keyframes. */
function SEM_applyOn(L, item, start) {
    var origId = SEM_origId(L);
    var orig = SEM_itemById(origId);
    if (SEM_MARK.test(L.comment || '')) SEM_restoreOn(L);      // re-process: start from the original
    var inP = L.inPoint, outP = L.outPoint, s = L.stretch / 100, nm = L.name;
    L.replaceSource(item, false);
    if (L.name !== nm) L.name = nm;          // keep the layer's name, not the render's file name
    var off = start / orig.frameRate;          // seconds of source the render skips at its head
    if (off > 0.00001) {
        // time-remap so render frame 0 sits on original frame `start` without moving the layer
        // (moving it would shift its keyframes)
        L.timeRemapEnabled = true;
        L.outPoint = outP; L.inPoint = inP;    // replaceSource may have collapsed in/out
        var tr = L.property('ADBE Time Remapping');
        tr.setValueAtTime(inP, (inP - L.startTime) / s - off);
        tr.setValueAtTime(outP, (outP - L.startTime) / s - off);
        var eps = 0.25 / L.containingComp.frameRate;
        for (var k = tr.numKeys; k >= 1; k--) {
            var kt = tr.keyTime(k);
            if (Math.abs(kt - inP) > eps && Math.abs(kt - outP) > eps) tr.removeKey(k);
        }
        for (k = 1; k <= tr.numKeys; k++)
            tr.setInterpolationTypeAtKey(k, KeyframeInterpolationType.LINEAR, KeyframeInterpolationType.LINEAR);
    }
    L.outPoint = outP; L.inPoint = inP;
    if (Math.abs(L.outPoint - outP) > 0.001) L.outPoint = outP;
    var c = (L.comment || '').replace(SEM_MARK, '').replace(/\s+$/, '');
    L.comment = (c ? c + '\n' : '') + '[SE Matte] orig=' + origId;
}

function SEM_import(argJson) {
    app.beginUndoGroup('SE Matte: remove background');
    var item = null, fresh = false;
    try {
        var a = eval('(' + argJson + ')');
        if (!a.prores) throw new Error('No render in the result.');
        var comp = SEM_findComp(a.ref.compId);
        if (!comp) return SEM_err('The composition "' + a.ref.compName + '" is gone. Render is in: ' + a.prores);
        var L = SEM_findLayer(comp, a.ref);
        if (!L) return SEM_err('Layer "' + a.ref.layerName + '" not found. Render is in: ' + a.prores);
        var prev = SEM_MARK.test(L.comment || '') ? L.source : null;   // cutout from an earlier run
        item = SEM_findImported(a.prores);
        if (!item) { item = SEM_importFile(a.prores, true); fresh = true; }
        SEM_applyOn(L, item, a.start);
        // tidy the project: drop the earlier cutout if nothing uses it any more (the file stays on disk)
        try {
            if (prev && prev !== item && prev.usedIn.length === 0 && prev.parentFolder && prev.parentFolder.name === 'SE Matte') prev.remove();
        } catch (e3) {}
        return SEM_json({ ok: true, layer: L.name });
    } catch (e) {
        try { if (item && fresh) item.remove(); } catch (e2) {}
        return SEM_err('Import failed: ' + e.toString());
    } finally {
        app.endUndoGroup();
    }
}

function SEM_restore(refJson) {
    app.beginUndoGroup('SE Matte: restore original');
    try {
        var ref = eval('(' + refJson + ')');
        var comp = SEM_findComp(ref.compId);
        var L = comp && SEM_findLayer(comp, ref);
        if (!L) return SEM_err('Layer not found.');
        if (!SEM_MARK.test(L.comment || '')) return SEM_err('This layer already shows its original footage.');
        SEM_restoreOn(L);
        return SEM_json({ ok: true });
    } catch (e) {
        return SEM_err('Restore failed: ' + e.toString());
    } finally {
        app.endUndoGroup();
    }
}
