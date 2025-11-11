// src/components/Diffeqn.jsx
import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  LogarithmicScale,
  CategoryScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  LogarithmicScale,
  CategoryScale,
  Tooltip,
  Legend
);

const PrettyPre = ({ children }) => (
  <pre
    style={{
      whiteSpace: "pre-wrap",
      backgroundColor: "#f8f9fa",
      border: "1px solid #e6e9ef",
      borderRadius: 8,
      padding: 10,
      fontSize: "0.9rem",
      marginBottom: 10,
      overflowX: "auto",
    }}
  >
    {children}
  </pre>
);

const Diffeqn = () => {
  const [n, setN] = useState(32);
  const [etaMax, setEtaMax] = useState(5.0);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState(null);

  const runSolver = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResp(null);
    try {
      const res = await fetch("http://localhost:5001/diffeqn_solver", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n: Number(n), eta_max: Number(etaMax) }),
      });
      const data = await res.json();
      if (data.status === "success") setResp(data.result);
      else setError(data.message || "Server error");
    } catch (err) {
      console.error(err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  // --- Chart 1: f_num vs f_exact ---
  const chartData1 =
    resp && {
      labels: resp.eta.map((v) => v.toFixed(3)),
      datasets: [
        {
          label: "Collocation Solution (f_num)",
          data: resp.f_num,
          borderColor: "rgba(0,123,255,0.9)",
          backgroundColor: "rgba(0,123,255,0.3)",
          tension: 0.3,
          pointRadius: 3,
          fill: false,
        },
        {
          label: "Analytical erf(η)",
          data: resp.f_exact,
          borderColor: "rgba(220,53,69,0.9)",
          backgroundColor: "rgba(220,53,69,0.3)",
          borderDash: [5, 5],
          tension: 0.3,
          pointRadius: 0,
          fill: false,
        },
      ],
    };

  // --- Chart 2: Error vs η ---
  const chartData2 =
    resp && {
      labels: resp.eta.map((v) => v.toFixed(3)),
      datasets: [
        {
          label: "Absolute Error |f_num - f_exact|",
          data: resp.error,
          borderColor: "rgba(255,165,0,0.9)",
          backgroundColor: "rgba(255,165,0,0.4)",
          tension: 0.3,
          pointRadius: 3,
          fill: false,
        },
      ],
    };

  const chartOptions = (logY = false) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      tooltip: { mode: "nearest" },
    },
    scales: {
      x: {
        title: { display: true, text: "η" },
        ticks: { font: { size: 12 } },
      },
      y: {
        title: {
          display: true,
          text: logY ? "Error (log scale)" : "f(η)",
        },
        type: logY ? "logarithmic" : "linear",
        ticks: {
          font: { size: 12 },
          callback: (val) => (logY ? val.toExponential(1) : val),
        },
      },
    },
  });

  return (
    <div className="container-fluid py-5 d-flex justify-content-center">
      <div style={{ width: "95%", maxWidth: "1600px" }}>
        <h2 className="text-center text-primary fw-bold mb-4">
          Gauss–Legendre Collocation ODE Solver
        </h2>

        <div className="card shadow p-4 mb-4 border-0 rounded-4">
          <form onSubmit={runSolver} className="row g-3 align-items-center">
            <div className="col-md-3">
              <label className="form-label fw-semibold">Nodes (n):</label>
              <input
                type="number"
                className="form-control"
                value={n}
                min={4}
                max={128}
                onChange={(e) => setN(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <label className="form-label fw-semibold">η_max:</label>
              <input
                type="number"
                className="form-control"
                value={etaMax}
                step="0.1"
                onChange={(e) => setEtaMax(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <button
                className="btn btn-primary w-100"
                type="submit"
                disabled={loading}
              >
                {loading ? "Running..." : "Run Solver"}
              </button>
            </div>
          </form>

          {!resp && !loading && (
            <div className="mt-4">
              <p className="text-secondary fw-semibold">
                This module performs Gauss–Legendre collocation for the ODE:
              </p>
              <PrettyPre>f'' + 2ηf' = 0, with f(0)=0, f(∞)=1</PrettyPre>
            </div>
          )}
        </div>

        {error && <div className="alert alert-danger">{error}</div>}

        {resp && (
          <div className="card shadow p-4 border-0 rounded-4">
            <h5 className="fw-bold text-dark mb-3">Results (n = {resp.n})</h5>

            <PrettyPre>
              {`η_max = ${resp.eta_max}\nMax abs error = ${resp.max_error.toExponential(
                4
              )}\nMean abs error = ${resp.mean_error.toExponential(4)}`}
            </PrettyPre>

            <h6 className="fw-semibold mt-3">Logs</h6>
            <PrettyPre>{resp.logs.join("\n")}</PrettyPre>

            <hr />
            <h5 className="fw-bold text-dark mb-3">f(η): Numerical vs Analytical</h5>
            <div
              style={{
                background: "#fff",
                padding: 25,
                borderRadius: 12,
                width: "100%",
                height: "500px",
              }}
            >
              <Line
                key={`chart1-${resp.n}-${resp.eta_max}`}
                data={chartData1}
                options={chartOptions(false)}
              />
            </div>

            <hr />
            <h5 className="fw-bold text-dark mb-3">Error vs η (Log Scale)</h5>
            <div
              style={{
                background: "#fff",
                padding: 25,
                borderRadius: 12,
                width: "100%",
                height: "500px",
              }}
            >
              <Line
                key={`chart2-${resp.n}-${resp.eta_max}`}
                data={chartData2}
                options={chartOptions(true)}
              />
            </div>

            <hr />
            <h6 className="fw-semibold mt-3">η Nodes</h6>
            <PrettyPre>
              [
              {resp.eta
                .map((v) => v.toFixed(3))
                .join(", ")}
              ]
            </PrettyPre>
          </div>
        )}
      </div>
    </div>
  );
};

export default Diffeqn;
