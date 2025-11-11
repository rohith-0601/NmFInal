// src/App.jsx
import { useState } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Harshad from "./components/HarshadPages/Harshad";
import Polynomial from "./components/PolynomialPages/Polynomial";
import GaussLeg from "./components/GuassPages/GuassLeg";
import Diffeqn from "./components/DiffeqnPages/Diffeqn";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      {/* Top Navigation Circles */}
      <div
        style={{
          position: "fixed",
          top: "20px",
          right: "30px",
          display: "flex",
          gap: "15px",
          zIndex: 1000,
        }}
      >
        <NavLink
          to="/"
          style={({ isActive }) => ({
            backgroundColor: isActive ? "#0d6efd" : "#e9ecef",
            color: isActive ? "white" : "black",
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textDecoration: "none",
            fontWeight: "600",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            transition: "all 0.2s ease-in-out",
          })}
        >
          H
        </NavLink>

        <NavLink
          to="/polynomial"
          style={({ isActive }) => ({
            backgroundColor: isActive ? "#0d6efd" : "#e9ecef",
            color: isActive ? "white" : "black",
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textDecoration: "none",
            fontWeight: "600",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            transition: "all 0.2s ease-in-out",
          })}
        >
          P
        </NavLink>

        <NavLink
          to="/gauss"
          style={({ isActive }) => ({
            backgroundColor: isActive ? "#0d6efd" : "#e9ecef",
            color: isActive ? "white" : "black",
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textDecoration: "none",
            fontWeight: "600",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            transition: "all 0.2s ease-in-out",
          })}
        >
          G
        </NavLink>

        <NavLink
          to="/diffeqn"
          style={({ isActive }) => ({
            backgroundColor: isActive ? "#0d6efd" : "#e9ecef",
            color: isActive ? "white" : "black",
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textDecoration: "none",
            fontWeight: "600",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            transition: "all 0.2s ease-in-out",
          })}
        >
          D
        </NavLink>
      </div>

      {/* Page Routes */}
      <Routes>
        <Route path="/" element={<Harshad />} />
        <Route path="/polynomial" element={<Polynomial />} />
        <Route path="/gauss" element={<GaussLeg />} />
        <Route path="/diffeqn" element={<Diffeqn />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
