function formatTable(rows) {
  if (!Array.isArray(rows) || rows.some((row) => !Array.isArray(row))) {
    throw new TypeError('array of arrays expected');
  }
  const cells = rows.map((row) => row.map((c) => (c === null || c === undefined ? '' : String(c))));
  const columns = Math.max(0, ...cells.map((row) => row.length));
  const widths = Array.from({ length: columns }, (_, i) => Math.max(...cells.map((row) => (row[i] === undefined ? 0 : row[i].length))));
  return cells
    .map((row) => widths.map((w, i) => (row[i] === undefined ? '' : row[i]).padEnd(w)).join(' | '))
    .join('\n');
}

module.exports = { formatTable };
