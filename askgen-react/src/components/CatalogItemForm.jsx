import React, { useEffect, useState, useRef } from 'react'

// Prefer the generic VITE_BACKEND_URL, then onprem, then cloud; fall back to 127.0.0.1:8000
const BACKEND = import.meta.env.VITE_BACKEND_URL || import.meta.env.VITE_BACKEND_ONPREM_URL || import.meta.env.VITE_BACKEND_CLOUD_URL || window.location.origin

export default function CatalogItemForm({ itemId, form, onDone }){
  const [item, setItem] = useState(null)
  const [values, setValues] = useState({})
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [lookupResults, setLookupResults] = useState({})
  const [lookupLoading, setLookupLoading] = useState({})
  const [displayValues, setDisplayValues] = useState({})
  const lookupTimers = useRef({})
  const [errors, setErrors] = useState({})
  const [lookupHighlight, setLookupHighlight] = useState({})

  useEffect(()=>{
    if(form){
      // backend provided full form payload
      const payloadItem = form.item
      // normalize variables: support array or object map
      let vars = form.variables || []
      if(!Array.isArray(vars) && vars && typeof vars === 'object') vars = Object.values(vars)
      // ensure each variable has a usable, unique `name` key for binding
      const seen = {}
      vars = vars.map((vv, idx)=>{
        const v = { ...vv }
        const base = v.name || v.element || v.variable_name || v.id || `var_${idx}`
        let name = base
        if(seen[name]){ name = `${base}_${idx}` }
        seen[base] = (seen[base] || 0) + 1
        v.name = name
        v.label = v.label || v.question_text || v.name || v.element || v.variable_name || v.id
        return v
      })
      setItem({ item: payloadItem, variables: vars })
      const initial = {}
      vars.forEach(v=> initial[v.name]=v.default || '')
      setValues(initial)
      return
    }
    if(!itemId) return
    setLoading(true)
    fetch(`${BACKEND}/catalog/item/${itemId}`).then(r=>r.json()).then(d=>{ const initial = {}; let vars = d.variables || []; if(!Array.isArray(vars) && vars && typeof vars === 'object') vars = Object.values(vars); // normalize names and labels
      const seen = {}
      vars = vars.map((vv, idx)=>{ const v = {...vv}; const base = v.name || v.element || v.variable_name || v.id || `var_${idx}`; let name = base; if(seen[name]){ name = `${base}_${idx}` } ; seen[base] = (seen[base]||0)+1; v.name = name; v.label = v.label || v.question_text || v.name || v.element || v.variable_name || v.id; return v })
      setItem({ item: d.item || d, variables: vars }); vars.forEach(v=> initial[v.name]=v.default || ''); setValues(initial) }).catch(e=>{ console.error(e); setItem(null) }).finally(()=>setLoading(false))
  }, [itemId, form])

  function setVal(name, val, label=null){
    setValues(v=>({ ...v, [name]: val }))
    setDisplayValues(d=>({ ...d, [name]: label !== null ? label : val }))
    // clear validation error for this field when user changes it
    setErrors(e=>{ const ne = {...e}; delete ne[name]; return ne })
  }

  function handleSelectLookup(v, r){
    // r = { value, label }
    setVal(v.name, r.value, r.label)
    setLookupResults(s=>({ ...s, [v.name]: [] }))
    setLookupHighlight(h=>{ const nh = {...h}; delete nh[v.name]; return nh })
  }

  function handleLookupKeyDown(e, v){
    const key = e.key
    const list = lookupResults[v.name] || []
    if(!list.length) return
    const cur = lookupHighlight[v.name] ?? -1
    if(key === 'ArrowDown'){
      e.preventDefault()
      const next = Math.min(cur+1, list.length-1)
      setLookupHighlight(h=>({ ...h, [v.name]: next }))
    }else if(key === 'ArrowUp'){
      e.preventDefault()
      const prev = Math.max(cur-1, 0)
      setLookupHighlight(h=>({ ...h, [v.name]: prev }))
    }else if(key === 'Enter'){
      e.preventDefault()
      const idx = cur >= 0 ? cur : 0
      const r = list[idx]
      if(r) handleSelectLookup(v, r)
    }else if(key === 'Escape'){
      setLookupResults(s=>({ ...s, [v.name]: [] }))
      setLookupHighlight(h=>{ const nh = {...h}; delete nh[v.name]; return nh })
    }
  }

  async function lookupField(v, q){
    try{
      const table = v.reference_table
      if(!table) return []
      const res = await fetch(`${BACKEND}/catalog/lookup?table=${encodeURIComponent(table)}&q=${encodeURIComponent(q)}`)
      if(!res.ok){
        console.error('lookup failed', res.status)
        setLookupResults(s=>({ ...s, [v.name]: [] }))
        return []
      }
      const data = await res.json()
      // Expect data.results = [{ label, value }, ...]
      setLookupResults(s=>({ ...s, [v.name]: data.results || [] }))
      return data.results || []
    }catch(e){ console.error(e); setLookupResults(s=>({ ...s, [v.name]: [] })); return [] }
  }

  // Triggered when user types into a lookup-enabled field. Debounces per-field.
  function handleLookupInput(v, q){
    // set display immediately
    setDisplayValues(d=>({ ...d, [v.name]: q }))
    // clear the current selected value until user picks an item
    setValues(s=>({ ...s, [v.name]: '' }))
    // clear any previous timer
    const timers = lookupTimers.current
    if(timers[v.name]){ clearTimeout(timers[v.name]) }
    timers[v.name] = setTimeout(async ()=>{
      setLookupLoading(l=>({ ...l, [v.name]: true }))
      const results = await lookupField(v, q)
      setLookupLoading(l=>({ ...l, [v.name]: false }))
      // if single result and user query exactly matches label, auto-select
      if(results && results.length===1){ const r = results[0]; setVal(v.name, r.value, r.label) }
    }, 350)
  }

  async function submitOrder(){
    setSubmitting(true)
    setResult(null)
    try{
      console.log('submitOrder start', { item, values, displayValues })
      // Client-side validation for mandatory fields
      const fldErrors = {}
      const varList = Array.isArray(item.variables) ? item.variables : (item.variables ? Object.values(item.variables) : [])
      varList.forEach(v=>{
        if(v.mandatory && (!values[v.name] || String(values[v.name]).trim()==='')){ fldErrors[v.name] = 'This field is required' }
      })
      if(Object.keys(fldErrors).length>0){ setErrors(fldErrors); setResult({ success:false, error: 'Validation failed' }); setSubmitting(false); return }

  const body = { item_id: item.item.sys_id, variables: values, display_values: displayValues, requested_for: undefined, comment: 'Ordered from AskGen UI' }
      const res = await fetch(`${BACKEND}/catalog/order`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
      if(!res.ok){
        // try to extract JSON error message
        let msg = `HTTP ${res.status}`
        try{ const j = await res.json(); msg = j.error || j.message || JSON.stringify(j) }catch(e){ const t = await res.text().catch(()=>null); if(t) msg = t }
        throw new Error(msg)
      }
  const data = await res.json()
  setResult({ success: true, data })
  try{ if(onDone && typeof onDone === 'function') onDone(data) }catch(e){ console.error('onDone handler failed', e) }
    }catch(e){ setResult({ success:false, error: e.message }) }
    finally{ setSubmitting(false) }
  }

  // Build a link to the created request/RITM if possible.
  // Heuristics (in order):
  // 1. response.request_url (explicit full url)
  // 2. response.request_sys_id + VITE_SN_INSTANCE_BASE_URL -> sc_req_item.do?sys_id=...
  // 3. env template VITE_SN_RITM_URL_TEMPLATE or VITE_INSTANCE_URL_TEMPLATE with {number} or {request_number}
  // 4. fallback to ServiceNow common pattern using VITE_SN_INSTANCE_BASE_URL and request_number
  function buildRequestLink(data){
    if(!data) return null
    // explicit full url from backend
    if(data.request_url) return data.request_url

    const base = import.meta.env.VITE_SN_INSTANCE_BASE_URL || import.meta.env.VITE_INSTANCE_BASE_URL || null
    // sys_id provided (prefer exact record link)
    if(data.request_sys_id && base){
      const b = base.replace(/\/$/, '')
      return `${b}/nav_to.do?uri=sc_req_item.do?sys_id=${data.request_sys_id}`
    }

    const tmpl = import.meta.env.VITE_SN_RITM_URL_TEMPLATE || import.meta.env.VITE_INSTANCE_URL_TEMPLATE || null
    if(data.request_number && tmpl){
      return tmpl.replace('{number}', data.request_number).replace('{request_number}', data.request_number).replace('{ritm}', data.request_number)
    }

    if(data.request_number && base){
      const b = base.replace(/\/$/, '')
      // navigate to request list filtered by number (ServiceNow common pattern)
      return `${b}/nav_to.do?uri=sc_request.do?sysparm_query=number=${encodeURIComponent(data.request_number)}`
    }

    return null
  }

  if(loading) return <div style={{padding:20}}>Loading item...</div>
  if(!item) return <div style={{padding:20}}>Item not found.</div>

  return (
    <div style={{padding:16}}>
      <h3>{item.item.name}</h3>
      <p>{item.item.short_description}</p>
      <div>
        {(item.variables||[]).map(v=> (
          <div key={v.id} style={{marginBottom:12}}>
            <label style={{display:'block',fontWeight:600}}>{v.label || v.name}{v.mandatory? ' *':''}</label>
            {v.type && (v.type==='choice' || v.type==='select') && v.choices ? (
              <select value={values[v.name]||''} onChange={e=>setVal(v.name, e.target.value)}>
                <option value="">-- select --</option>
                {v.choices.map(c=> <option key={c.value} value={c.value}>{c.label}</option>)}
              </select>
            ) : (
              <div style={{display:'flex',alignItems:'center',gap:8,flexDirection:'column'}}>
                {v.lookup ? (
                  <div style={{display:'flex',alignItems:'center',gap:8,width:'100%'}}>
                    <input
                      value={displayValues[v.name] ?? values[v.name] ?? ''}
                      onChange={e=>handleLookupInput(v, e.target.value)}
                      placeholder={v.hint || ''}
                      style={{flex:1}}
                    />
                    <button title="Search" onClick={async ()=>{ const q = displayValues[v.name]||''; setLookupLoading(l=>({ ...l, [v.name]: true })); const res = await lookupField(v, q); setLookupLoading(l=>({ ...l, [v.name]: false })); if(res && res.length===1){ setVal(v.name, res[0].value, res[0].label) } }}>🔍</button>
                    {lookupLoading[v.name] ? <div style={{paddingLeft:6}}>…</div> : null}
                  </div>
                ) : (
                  <input value={values[v.name]||''} onChange={e=>setVal(v.name, e.target.value)} />
                )}
                {/* render lookup results if present */}
                {lookupResults[v.name] && lookupResults[v.name].length>0 ? (
                  <div style={{border:'1px solid #ddd',padding:8,marginTop:6,width:'100%'}}>
                    {lookupResults[v.name].map((r, idx)=> (
                      <div
                        key={r.value}
                        style={{padding:6,cursor:'pointer', background: (lookupHighlight[v.name]===idx? '#eef' : 'transparent')}}
                        onMouseEnter={()=> setLookupHighlight(h=>({ ...h, [v.name]: idx }))}
                        onMouseLeave={()=> setLookupHighlight(h=>{ const nh = {...h}; delete nh[v.name]; return nh })}
                        onClick={()=>{ setVal(v.name, r.value, r.label); setLookupResults(s=>({ ...s, [v.name]: [] })) }}
                      >{r.label}</div>
                    ))}
                  </div>
                ) : null}
              </div>
            )}
            {errors[v.name] ? (<div style={{color:'red',fontSize:12,marginTop:6}}>{errors[v.name]}</div>) : null}
          </div>
        ))}
      </div>
      <div style={{marginTop:12}}>
        <button onClick={submitOrder} disabled={submitting}>{submitting? 'Ordering...':'Place Order'}</button>
        <button onClick={onDone} style={{marginLeft:8}}>Back</button>
        <button onClick={()=>window.dispatchEvent(new CustomEvent('askgen:catalog:conversational',{detail: item.item.sys_id}))} style={{marginLeft:8}}>Let the bot ask me</button>
      </div>
      {result && (
        <div style={{marginTop:12}}>
          {result.success ? (
            <div>
              <div style={{fontWeight:600}}>{result.data?.message || 'Order placed'}</div>
              {result.data?.request_number ? (
                <div style={{marginTop:6}}>Request number: <strong>{result.data.request_number}</strong></div>
              ) : null}
              {(() => {
                const link = buildRequestLink(result.data)
                // If backend returned explicit RITMs array, render them as a list with links
                if(result.data && Array.isArray(result.data.ritms) && result.data.ritms.length>0){
                  return (
                    <div style={{marginTop:8}}>
                      <div style={{fontWeight:600, marginBottom:6}}>Created request items</div>
                      <ul>
                        {result.data.ritms.map((r, idx)=> (
                          <li key={r.sys_id || r.number || idx} style={{marginBottom:6}}>
                            {r.number ? (<strong>{r.number}</strong>) : (r.sys_id ? (<span>RITM {r.sys_id}</span>) : null)}
                            {r.short_description ? (<span style={{marginLeft:8}}>- {r.short_description}</span>) : null}
                            {r.url ? (
                              <div><a href={r.url} target="_blank" rel="noopener noreferrer">Open RITM</a></div>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                      {/* also expose a top-level request link when available */}
                      {link ? (<div style={{marginTop:6}}><a href={link} target="_blank" rel="noopener noreferrer">Open parent request</a></div>) : null}
                    </div>
                  )
                }
                // otherwise fall back to a single request link or raw JSON
                if(link) return (<div style={{marginTop:8}}><a href={link} target="_blank" rel="noopener noreferrer">Open request in instance</a></div>)
                return (<pre>{JSON.stringify(result.data,null,2)}</pre>)
              })()}
            </div>
          ) : <div style={{color:'red'}}>{result.error}</div>}
        </div>
      )}
    </div>
  )
}
