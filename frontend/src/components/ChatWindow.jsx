import { useRef, useEffect, useState } from 'react'
import ProductCard from './ProductCard'
import ProductDetail from './ProductDetail'

// Skeleton placeholder cards shown while waiting for search results
function SkeletonGrid() {
  return (
    <div className="product-grid">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="skeleton-card" style={{ '--card-i': i }} />
      ))}
    </div>
  )
}

export default function ChatWindow({ messages, loading }) {
  const endRef = useRef(null)
  const [selectedProduct, setSelectedProduct] = useState(null)

  // Auto-scroll to the newest message whenever messages or loading changes
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Show skeleton cards when loading AND the last user message suggests a search
  const showSkeleton = loading

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`row ${msg.role}`}>
            <div className={`bubble ${msg.role} ${msg.error ? 'error' : ''}`}>
              {msg.content}
            </div>

            {msg.role === 'assistant' && msg.products?.length > 0 && (
              <div className="product-grid">
                {msg.products.map((p, j) => (
                  <ProductCard
                    key={j}
                    product={p}
                    style={{ '--card-i': j }}
                    onClick={(product) => setSelectedProduct(product)}
                  />
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="row assistant">
            <div className="bubble assistant typing">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
            {showSkeleton && <SkeletonGrid />}
          </div>
        )}

        <div ref={endRef} />
      </div>

      {/* Product Detail Modal */}
      {selectedProduct && (
        <ProductDetail
          product={selectedProduct}
          onClose={() => setSelectedProduct(null)}
        />
      )}
    </div>
  )
}
