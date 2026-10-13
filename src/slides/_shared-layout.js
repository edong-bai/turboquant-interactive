// Layout helper shared by the rotation / sphere / concentration slides: inline CSS grids that
// collapse to one column below 900px (matching .stage in styles.css).

const NARROW = '(max-width: 900px)';

/**
 * Make `grid` a CSS grid with `wideCols` on wide screens and a single column on narrow ones.
 * `vizzes` (canvas containers) fill their grid cell when wide and get a fixed height when narrow.
 * When `fill` is true the grid takes the remaining slide height on wide screens.
 */
export function responsiveGrid(grid, wideCols, { vizzes = [], fill = false, narrowHeight = 'clamp(280px, 52vh, 600px)' } = {}) {
  const mq = window.matchMedia(NARROW);
  grid.style.display = 'grid';
  grid.style.minHeight = '0';
  const apply = () => {
    const narrow = mq.matches;
    grid.style.gridTemplateColumns = narrow ? 'minmax(0, 1fr)' : wideCols;
    if (fill) grid.style.flex = narrow ? '0 0 auto' : '1 1 0';
    for (const v of vizzes) {
      v.style.height = narrow ? narrowHeight : '100%';
      v.style.minHeight = '0';
    }
  };
  mq.addEventListener('change', apply);
  apply();
  return grid;
}
