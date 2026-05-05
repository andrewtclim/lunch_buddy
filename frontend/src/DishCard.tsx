type DishCardProps = {
  dish_name: string;
  dining_hall: string;
  reason: string;
  distance_m: number | null;
  picked: boolean;
  disabled: boolean;
  onPick: () => void;
};

function formatDistance(m: number | null): string | null {
  if (m === null) return null;
  const ft = Math.round(m * 3.281);
  return ft < 5280 ? `${ft}ft` : `${(ft / 5280).toFixed(1)}mi`;
}

export default function DishCard({
  dish_name,
  dining_hall,
  reason,
  distance_m,
  picked,
  disabled,
  onPick,
}: DishCardProps) {
  const dist = formatDistance(distance_m);

  return (
    <div className="dish-card">
      <h3 className="dish-card__name">{dish_name}</h3>
      <p className="dish-card__hall">
        {dining_hall}
        {dist && <span className="dish-card__dist"> · {dist}</span>}
      </p>
      <p className="dish-card__reason">{reason}</p>
      <button
        type="button"
        className="dish-card__pick"
        disabled={disabled}
        onClick={onPick}
      >
        {picked ? "Picking..." : "Pick this"}
      </button>
    </div>
  );
}
