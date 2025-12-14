import React from 'react'

export default function CatalogList({ items, loading, onSelect }){
  if(loading) return <div style={{padding:12}}>Loading...</div>
  if(!items || items.length===0) return <div style={{padding:12}}>No items found.</div>
  return (
    <div style={{padding:8}}>
      {items.map(it=> (
        <div key={it.sys_id} className="catalog-item" onClick={()=>onSelect(it.sys_id)}>
          <div className="catalog-item-title">{it.name}</div>
          <div className="catalog-item-desc">{it.short_description}</div>
        </div>
      ))}
    </div>
  )
}
