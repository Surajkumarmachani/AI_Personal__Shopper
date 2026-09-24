import { useState } from 'react'

export default function ProductCard({ product, onClick, style }) {
  const [imgOk, setImgOk] = useState(true)
  const { title, brand, price, currency = 'USD', image_url, category } = product

  const priceLabel =
    price != null ? `${currency === 'USD' ? '$' : ''}${price}` : '—'

  return (
    <div
      className="product-card"
      style={style}
      tabIndex={0}
      role="button"
      aria-label={`View details for ${title}`}
      onClick={() => onClick && onClick(product)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick?.(product) } }}
    >
      <div className="product-image">
        {imgOk && image_url ? (
          <img
            src={image_url}
            alt={title}
            loading="lazy"
            onError={() => setImgOk(false)}
          />
        ) : (
          <div className="image-fallback">
            {(brand || title || '?').charAt(0).toUpperCase()}
          </div>
        )}
        {category && <span className="category-tag">{category}</span>}
      </div>

      <div className="product-info">
        <div className="product-title" title={title}>
          {title}
        </div>
        <div className="product-meta">
          {brand && <span className="product-brand">{brand}</span>}
          <span className="product-price">{priceLabel}</span>
        </div>
        <div className="product-link-row">
          <span className="product-link-text">View Details →</span>
        </div>
      </div>
    </div>
  )
}
