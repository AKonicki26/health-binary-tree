import './App.css'
import MetadataTable from "./Components/Metadata/MetadataTable.tsx";
import Visualization from "./Components/Visualization/Visualization.tsx";
import Predictor from "./Components/Predictor/Predictor.tsx";

function App() {

  return (
    <>
      <h1>COVID-19 Health Dashboard</h1>

      {/* COVID-19 Risk Predictor */}
      <Predictor />
      
      <Visualization 
        imagePath="/bp-analysis.png" 
        title="Blood Pressure Analysis"
        description="Distribution of COVID-19 cases across different blood pressure ranges. The data shows relatively consistent positive rates across all ranges."
      />

      <Visualization
        imagePath="/age-analysis.png"
        title="Age Analysis"
        description="Distribution of COVID-19 survey respondents. The data shows a relatively even distribution of respondents"
      />

      <Visualization
        imagePath="/cmbd-analysis.png"
        title="Blood Pressure Analysis"
        description="Distribution of COVID-19 cases across different heart rate ranges. The data shows relatively consistent positive rates across all ranges."
      />

      <Visualization
        imagePath="/hr-analysis.png"
        title="Blood Pressure Analysis"
        description="Distribution of COVID-19 cases across different comorbidity ranges. The data shows relatively consistent positive rates across all ranges."
      />

      <MetadataTable csvPath="/metadata.csv" title="Dataset Statistics" />
    </>
  )
}

export default App
