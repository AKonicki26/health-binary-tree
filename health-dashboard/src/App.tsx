import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import MetadataTable from "./Components/Metadata/MetadataTable.tsx";
import Visualization from "./Components/Visualization/Visualization.tsx";
import Predictor from "./Components/Predictor/Predictor.tsx";

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>COVID-19 Health Dashboard</h1>
      
      {/* COVID-19 Risk Predictor */}
      <Predictor />
      
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      
      <Visualization 
        imagePath="/bp-analysis.png" 
        title="Blood Pressure Analysis"
        description="Distribution of COVID-19 cases across different blood pressure ranges. The data shows relatively consistent positive rates across all ranges."
      />
      
      <MetadataTable csvPath="/metadata.csv" title="Dataset Statistics" />
      
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
