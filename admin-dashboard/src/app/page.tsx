"use client";
import { useEffect, useState } from 'react'

type RowType = {
  timestamp: string,
  operatorId: string,
  operatorName: string,
  activityName: string,
  gravity: number,
  raspberryName: string,
}

export default function HomePage() {
  const [rows, setRows] = useState<RowType[]>([]);
  const [lastRefresh, setLastRefresh] = useState("");
  // timestamp operatorId activity gravity
  useEffect(() => {
    setInterval(async () => {
      const last = await fetch('/api/notifications');
      const lastJson = await last.json();
      setRows(lastJson.rows);
      setLastRefresh(lastJson.lastRefresh);
      console.log(lastJson);
    }, 3000);
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
      <h1 className="text-4xl font-bold">Admin Dashboard</h1>
      {/* {typeof(lastRefresh) == String ? lastRefresh : lastRefresh.toISOString()} */}
      {lastRefresh}
      <table className="px-6 py-4 whitespace-nowrap text-sm text-yellow mt-5 border-2 rounded-xl bg-gray-100 text-black">
        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
          Time
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
          Operator Name
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
          Activity
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
          Gravity
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
          Raspberry
        </td>
    
        {
          rows.map(
            (row) => { return (<tr>
              <td className="px-6 py-4 whitespace-nowrap text-sm ">
                {row.timestamp}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm ">
                {row.operator.operatorName}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm ">
                {row.activityName}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm ">
                {row.gravity}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm ">
                {row.raspberryName}
              </td>
              </tr>)}
          )
        }
      </table>
    </main>
  );
}
