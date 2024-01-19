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

type OperatorStatus = {
  operatorName: string,
  lastUpdate: string,
  status: string
}

export default function HomePage() {
  const [rows, setRows] = useState<RowType[]>([]);
  const [lastRefresh, setLastRefresh] = useState("");

  const [operatorsStatus, setOperatorsStatus] = useState<OperatorStatus[]>([]);
  // timestamp operatorId activity gravity
  useEffect(() => {
    setInterval(async () => {
      const last = await fetch('/api/notifications');
      const lastJson = await last.json();
      setRows(lastJson.rows);
      setLastRefresh(lastJson.lastRefresh);
      
      const operators = await fetch('/api/live');
      const operatorsJson = await operators.json();
      setOperatorsStatus(operatorsJson.operators);
      console.log(operatorsJson);
    }, 3000);
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
      <h1 className="text-4xl font-bold m-6">Admin Dashboard</h1>
      {/* {typeof(lastRefresh) == String ? lastRefresh : lastRefresh.toISOString()} */}
      {lastRefresh}

      <div className='flex gap-24 '>

      <div className='mt-5 px-6 py-4 bg-gray-100 w-[16rem] text-black rounded-2xl shadow-xl'>
        <h3>Operators Live</h3>
        {
          operatorsStatus.map(
            (operator) => { return (<div className="px-6 py-2 whitespace-nowrap text-sm text-yellow mt-1 border-2 rounded-xl bg-gray-100 text-black">
              <p className="px-6 first-letter:whitespace-nowrap text-sm font-bold">
                {operator.operatorName}
              </p>
              <p className="px-6 whitespace-nowrap text-sm font-bold">
                {operator.status}
                
              </p>
              <p className="px-6 whitespace-nowrap text-sm font-bold">
                {operator.lastUpdate}
              </p>
            </div>)}
          )
        }
      </div>

      <table className="px-10 py-4 whitespace-nowrap text-sm text-yellow mt-5 border-2 rounded-xl bg-gray-100 text-black">
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
      </div>
    </main>
  );
}
