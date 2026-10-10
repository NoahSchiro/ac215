import type { NormalizedPoint } from './points'

interface Props {
  point: NormalizedPoint
}

/** A target marker positioned by normalized viewport coordinates. */
export function Crosshair({ point }: Props) {
  return (
    <div
      className="crosshair"
      data-testid="crosshair"
      aria-hidden="true"
      style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
    >
      <span className="crosshair-ring" />
      <span className="crosshair-line horizontal" />
      <span className="crosshair-line vertical" />
    </div>
  )
}
