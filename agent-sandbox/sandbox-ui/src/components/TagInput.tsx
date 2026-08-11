import { useState } from 'react'

export interface TagInputProps {
  label: string
  value: string[]
  onChange: (tags: string[]) => void
  hint?: string
}

export function TagInput({ label, value, onChange, hint }: TagInputProps) {
  const [draft, setDraft] = useState('')

  function commit() {
    const trimmed = draft.trim()
    if (trimmed && !value.includes(trimmed)) onChange([...value, trimmed])
    setDraft('')
  }

  return (
    <label>
      {label} {hint && <span className="hint">— {hint}</span>}
      <div className="tag-input">
        {value.map((tag) => (
          <span className="tag" key={tag}>
            {tag}
            <button type="button" onClick={() => onChange(value.filter((t) => t !== tag))} aria-label={`remove ${tag}`}>
              ×
            </button>
          </span>
        ))}
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ',') {
              e.preventDefault()
              commit()
            } else if (e.key === 'Backspace' && draft === '' && value.length > 0) {
              onChange(value.slice(0, -1))
            }
          }}
          onBlur={commit}
          placeholder="tool name, press Enter"
        />
      </div>
    </label>
  )
}
