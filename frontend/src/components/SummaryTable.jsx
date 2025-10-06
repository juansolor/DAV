import React from 'react';

export default function SummaryTable({ summary }) {
  const stats = Object.keys(summary[Object.keys(summary)[0]] || {});
  const columns = Object.keys(summary);
  return (
    <table border="1" cellPadding="6" style={{borderCollapse: 'collapse', marginTop: 12}}>
      <thead>
        <tr>
          <th>Métrica</th>
          {columns.map(c => <th key={c}>{c}</th>)}
        </tr>
      </thead>
      <tbody>
        {stats.map(stat => (
          <tr key={stat}>
            <td><strong>{stat}</strong></td>
            {columns.map(c => (
              <td key={c+stat}>{
                typeof summary[c][stat] === 'number' ? summary[c][stat].toFixed(3) : summary[c][stat]
              }</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
