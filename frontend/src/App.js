import "@/App.css";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import MainLayout from "@/components/layout/MainLayout";

// Original Pages (Enhanced)
import Dashboard from "./pages/Dashboard";
import Analytics from "./pages/Analytics";
import Recommendations from "./pages/Recommendations";

// New Pages
import Landing from "./pages/Landing";
import CommandCenter from "./pages/CommandCenter";
import GlobeExplorer from "./pages/GlobeExplorer";
import NetworkVisualizer from "./pages/NetworkVisualizer";
import AirportIntelligence from "./pages/AirportIntelligence";
import AirlineIntelligence from "./pages/AirlineIntelligence";
import CountryIndex from "./pages/CountryIndex";
import MLLab from "./pages/MLLab";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/landing" element={<Landing />} />
        
        {/* Main Application with Sidebar */}
        <Route path="/*" element={
          <MainLayout>
            <Routes>
              <Route path="/" element={<CommandCenter />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/recommendations" element={<Recommendations />} />
              
              <Route path="/globe" element={<GlobeExplorer />} />
              <Route path="/network" element={<NetworkVisualizer />} />
              <Route path="/airports" element={<AirportIntelligence />} />
              <Route path="/airlines" element={<AirlineIntelligence />} />
              <Route path="/countries" element={<CountryIndex />} />
              <Route path="/ml-lab" element={<MLLab />} />
              
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </MainLayout>
        } />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
