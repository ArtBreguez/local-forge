// parseCsvLine(line) -> array of field strings.
// Fields are separated by commas. A field may be wrapped in double quotes; then it may contain commas, and a
// doubled quote ("") inside it stands for one literal quote character. A trailing comma means a last empty
// field. An empty line gives [''].
function parseCsvLine(line) {
  const fields = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      fields.push(field);
      field = '';
    } else {
      field += ch;
    }
  }
  fields.push(field);
  return fields;
}

module.exports = { parseCsvLine };
