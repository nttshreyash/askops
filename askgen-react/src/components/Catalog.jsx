import React, { useEffect, useState } from 'react'
import CatalogList from './CatalogList'
import CatalogItemForm from './CatalogItemForm'

const BACKEND = (import.meta.env.VITE_BACKEND_ONPREM_URL) || ''

export default function Catalog(){
  const [items, setItems] = useState([])
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(false)

  async function fetchItems(q=''){
    setLoading(true)
    try{
      const res = await fetch(`${BACKEND}/catalog/items?query=${encodeURIComponent(q)}&page=${page}&limit=25`)
      if(!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setItems(data.items || [])
    }catch(e){
      console.error('Catalog list error', e)
      setItems([])
    }finally{ setLoading(false) }
  }

  useEffect(()=>{ fetchItems(query) }, [page])

  return (
    <div className="catalog-root">
      <div className="catalog-sidebar">
        <div style={{padding:8}}>
          <input placeholder="Search catalog" value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>{ if(e.key==='Enter'){ setPage(1); fetchItems(e.target.value) } }} />
          <button onClick={()=>{ setPage(1); fetchItems(query) }} style={{marginLeft:8}}>Search</button>
        </div>
        <CatalogList items={items} loading={loading} onSelect={setSelected} />
      </div>
      <div className="catalog-main">
        {selected ? <CatalogItemForm itemId={selected} onDone={()=>{ setSelected(null); fetchItems(query) }} /> : <div style={{padding:20}}>Select a catalog item to view details and order.</div>}
      </div>
    </div>
  )
}
