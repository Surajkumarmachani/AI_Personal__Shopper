import { useState } from 'react'

// Helper to render star ratings like Amazon
function StarRating({ rating, count }) {
  if (!rating) return null
  const fullStars = Math.floor(rating)
  const hasHalf = rating - fullStars >= 0.5
  const emptyStars = 5 - fullStars - (hasHalf ? 1 : 0)

  return (
    <div className="amazon-ratings">
      <div className="amazon-stars" title={`${rating} out of 5 stars`}>
        {Array(fullStars).fill('★').map((_, i) => <span key={`f${i}`} className="star full">★</span>)}
        {hasHalf && <span className="star half">★</span>}
        {Array(emptyStars).fill('★').map((_, i) => <span key={`e${i}`} className="star empty">★</span>)}
        <i className="amazon-chevron"></i>
      </div>
      {count > 0 && <span className="amazon-rating-count">{count.toLocaleString()} ratings</span>}
    </div>
  )
}

export default function ProductDetail({ product, onClose }) {
  const [imgOk, setImgOk] = useState(true)
  if (!product) return null

  const { title, brand, price, currency = 'USD', image_url, category, description, product_url, avg_rating, rating_count } = product

  const priceLabel = price != null ? `${currency === 'USD' ? '$' : ''}${price}` : 'Currently unavailable'

  return (
    <div className="amazon-modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="Product details">
      <div className="amazon-modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Header / Nav */}
        <div className="amazon-modal-header">
          <div className="amazon-breadcrumb">
            {category || 'Departments'} › {brand || 'Brand'}
          </div>
          <button className="amazon-close-btn" onClick={onClose} aria-label="Close product details">✕</button>
        </div>

        <div className="amazon-modal-body">
          {/* Left Column: Image */}
          <div className="amazon-image-col">
            {imgOk && image_url ? (
              <img
                src={image_url}
                alt={title}
                className="amazon-main-image"
                onError={() => setImgOk(false)}
              />
            ) : (
              <div className="amazon-image-fallback">
                {(brand || title || '?').charAt(0).toUpperCase()}
              </div>
            )}
          </div>

          {/* Center Column: Core Details */}
          <div className="amazon-details-col">
            <h1 className="amazon-title">{title}</h1>
            {brand && <a href="#" className="amazon-brand-link">Brand: {brand}</a>}
            
            <StarRating rating={avg_rating} count={rating_count} />

            <hr className="amazon-divider" />

            <div className="amazon-price-block">
              {price != null && <span className="amazon-price-symbol">$</span>}
              <span className="amazon-price-whole">{price != null ? Math.floor(price) : 'Currently unavailable'}</span>
              {price != null && <span className="amazon-price-fraction">{(price % 1).toFixed(2).substring(2)}</span>}
            </div>
            {price != null && <div className="amazon-returns">Free Returns</div>}

            <hr className="amazon-divider" />

            {/* Simulated Amazon Product Details section */}
            <div className="amazon-product-details-box">
              <h3 className="amazon-section-title">Product details</h3>
              <table className="amazon-details-table">
                <tbody>
                  {category && (
                    <tr>
                      <th className="amazon-th">Department</th>
                      <td className="amazon-td">{category}</td>
                    </tr>
                  )}
                  {brand && (
                    <tr>
                      <th className="amazon-th">Manufacturer</th>
                      <td className="amazon-td">{brand}</td>
                    </tr>
                  )}
                  <tr>
                    <th className="amazon-th">ASIN</th>
                    <td className="amazon-td">{product.id || 'N/A'}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <hr className="amazon-divider" />

            {/* Description as "About this item" */}
            <div className="amazon-about-item">
              <h3 className="amazon-section-title">About this item</h3>
              <ul className="amazon-feature-list">
                {/* We fake a bullet list by splitting on periods for long descriptions */}
                {description ? (
                  description.split('. ').filter(s => s.length > 10).slice(0, 5).map((point, i) => (
                    <li key={i}>{point.trim() + (point.endsWith('.') ? '' : '.')}</li>
                  ))
                ) : (
                  <li>No description available.</li>
                )}
              </ul>
            </div>
          </div>

          {/* Right Column: Buy Box */}
          <div className="amazon-buybox-col">
            <div className="amazon-buybox-card">
              <div className="amazon-price-block">
                {price != null && <span className="amazon-price-symbol">$</span>}
                <span className="amazon-price-whole">{price != null ? Math.floor(price) : 'N/A'}</span>
                {price != null && <span className="amazon-price-fraction">{(price % 1).toFixed(2).substring(2)}</span>}
              </div>
              <div className="amazon-delivery">
                <span className="amazon-prime-icon">✓ prime</span>
                <p>FREE delivery <strong>Tomorrow</strong>.</p>
              </div>
              
              <div className="amazon-stock-status">In Stock</div>

              {product_url && (
                <a
                  href={product_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="amazon-btn-primary"
                >
                  View on Amazon
                </a>
              )}

              <div className="amazon-secure-transaction">
                🔒 Secure transaction
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
