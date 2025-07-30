'use client'
import { useEffect, useState } from "react";
import HistogramChart from './graph.js';
import { redirect } from "next/dist/server/api-utils/index.js";
import { useRouter } from 'next/navigation.js'

 

export default function Home() {
  const [res, setRes] = useState("No data yet");
  const [loading, setLoading] = useState(false);
  const [steps, setSteps] = useState([]);
  const [dates, setDates] = useState(["6-17-25", "6-18-25", "6-19-25", "6-20-25"]);
  const [today, setToday] = useState(null);
  const router = useRouter();
  const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://lifestyle-api.onrender.com/'
  : 'http://localhost:5000';
  
const getAuth = async () =>{
 router.push('/auth')



  
}
  const getData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/endpoint`);
      const data = await response.json();
      console.log(data.evaluation);
      setRes(data.evaluation);
      setSteps(data.stepCount);
      setDates(data.dates);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setRes({ error: "Failed to fetch data" });
      setLoading(false);
    }
  };

  useEffect(() => {
    const now = new Date();
    setToday(now.toLocaleDateString());  

  }, [loading]);
  // const chartData = dates.map((date, index) => ({
  //   date,
  //   steps: steps[index] || 0  
  // }));

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 py-10 gap-8 font-[family-name:var(--font-geist-sans)]">
      
      <div className="text-center space-y-2">
        <p className="text-xl font-semibold">Welcome !</p>
        <p className="text-gray-600">Today&apos;s date: {today}</p>
      </div>

      <div className="w-full max-w-3xl space-y-4">
        <pre className="p-4 rounded text-left whitespace-pre-wrap">
          {typeof res === 'string' ? res : JSON.stringify(res, null, 2)}
        </pre>

        {loading && (
          <p className="text-blue-500 font-medium">Please wait for the data to load...</p>
        )}

        <div className="w-full">
       
    
      </div>
      </div>

      <button
        className="mt-6 bg-blue-500 text-white px-5 py-2 rounded hover:bg-blue-600 transition"
        onClick={getData}
      >
        Fetch Data
      </button>

      <button
        className="mt-6 bg-blue-500 text-white px-5 py-2 rounded hover:bg-blue-600 transition"
        onClick={getAuth}
      >
       Authorize
      </button>
    </div>
  );
}


