import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

const HistogramChart = ({ data, title }) => (
  <div style={{ padding: '16px' ,text:'white'}}>
    <h2>{title}</h2>
    <div style={{ width: 600 }}>
      <LineChart stroke="#ccc" width={600} height={400} data={data}>
        <CartesianGrid  strokeDasharray="3 3" stroke="#ccc" />
        <XAxis  stroke="#ccc" dataKey="bin" />
        <YAxis stroke="#ccc" />
        <Tooltip />
        <Line  dataKey="count" fill="#8884d8" />
      </LineChart>
    </div>
  </div>
);

export default HistogramChart;