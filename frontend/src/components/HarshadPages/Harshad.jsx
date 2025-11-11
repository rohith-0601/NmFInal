import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

const Harshad = () => {
  // -------------------------------
  // Question A States
  // -------------------------------
  const [q1Start, setQ1Start] = useState("");
  const [q1End, setQ1End] = useState("");
  const [q1Result, setQ1Result] = useState(null);
  const [q1Loading, setQ1Loading] = useState(false);

  const handleQ1Submit = async (e) => {
    e.preventDefault();
    setQ1Loading(true);
    setQ1Result(null);
    try {
      const res = await fetch("http://localhost:5001/first_non_harshad", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start: Number(q1Start), end: Number(q1End) }),
      });
      const data = await res.json();
      setQ1Result(data);
    } catch (err) {
      console.error(err);
    } finally {
      setQ1Loading(false);
    }
  };

  // -------------------------------
  // Question B States
  // -------------------------------
  const [mode, setMode] = useState(null);
  const [startRange, setStartRange] = useState("");
  const [endRange, setEndRange] = useState("");
  const [targetCount, setTargetCount] = useState("");
  const [q2Result, setQ2Result] = useState(null);
  const [q2Loading, setQ2Loading] = useState(false);

  const handleModeSelect = (m) => {
    setMode(m);
    setQ2Result(null);
  };

  const handleQ2Submit = async (e) => {
    e.preventDefault();
    setQ2Loading(true);
    setQ2Result(null);
    try {
      let payload = {};
      if (mode === 1) {
        payload = {
          mode: 1,
          start_range: Number(startRange),
          end_range: Number(endRange),
        };
      } else {
        payload = { mode: 2, target_count: Number(targetCount) };
      }

      const res = await fetch("http://localhost:5001/harshad_groups", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setQ2Result(data);
    } catch (err) {
      console.error(err);
    } finally {
      setTimeout(() => setQ2Loading(false), 500);
    }
  };

  // -------------------------------
  // UI
  // -------------------------------
  return (
    <div className="container py-5">
      <h2 className="text-center mb-5 text-primary fw-bold">Harshad Numbers</h2>

      {/* ==================== QUESTION A ==================== */}
      <div className="card shadow-lg p-4 mb-5 border-0 rounded-4">
        <h4 className="text-dark fw-semibold mb-3">
          A. Find the first factorial which is not a Harshad number
        </h4>
        <form onSubmit={handleQ1Submit}>
          <div className="row mb-3">
            <div className="col-md-5">
              <input
                type="number"
                className="form-control"
                placeholder="Enter start number"
                value={q1Start}
                onChange={(e) => setQ1Start(e.target.value)}
                required
              />
            </div>
            <div className="col-md-5">
              <input
                type="number"
                className="form-control"
                placeholder="Enter end number"
                value={q1End}
                onChange={(e) => setQ1End(e.target.value)}
                required
              />
            </div>
            <div className="col-md-2">
              <button type="submit" className="btn btn-primary w-100">
                Find
              </button>
            </div>
          </div>
        </form>

        {q1Loading && (
          <div className="text-center my-3">
            <div
              className="spinner-border text-primary"
              role="status"
              style={{ width: "3rem", height: "3rem" }}
            ></div>
            <p className="mt-2 text-secondary fw-semibold">
              Processing... Please wait
            </p>
          </div>
        )}

        {q1Result && !q1Loading && (
          <div className="mt-4 bg-light p-3 rounded">
            {q1Result.status === "non-harshad-found" ? (
              <>
                <h5 className="text-success fw-semibold">
                  ❌ First Non-Harshad Number: {q1Result.number}
                </h5>
                <p>
                  <strong>Digit Sum:</strong> {q1Result.digit_sum} <br />
                  <strong>Remainder:</strong> {q1Result.remainder}
                </p>
                <p>
                  <strong>Factorial:</strong> <br />
                  <span
                    style={{
                      wordBreak: "break-all",
                      color: "#333",
                      fontSize: "0.9rem",
                    }}
                  >
                    {q1Result.factorial}
                  </span>
                </p>
              </>
            ) : (
              <p className="text-info">{q1Result.message}</p>
            )}
          </div>
        )}
      </div>

      {/* ==================== QUESTION B ==================== */}
      <div className="card shadow-lg p-4 border-0 rounded-4">
        <h4 className="text-dark fw-semibold mb-3">
          B. Find consecutive Harshad numbers
        </h4>
        <p className="text-secondary">
          Example: 110, 111, 112 are three consecutive Harshad numbers.
        </p>

        <div className="d-flex gap-3 mb-4">
          <button
            type="button"
            className={`btn ${
              mode === 1 ? "btn-primary" : "btn-outline-primary"
            } w-50`}
            onClick={() => handleModeSelect(1)}
          >
            Mode 1 – Range of Groups
          </button>
          <button
            type="button"
            className={`btn ${
              mode === 2 ? "btn-primary" : "btn-outline-primary"
            } w-50`}
            onClick={() => handleModeSelect(2)}
          >
            Mode 2 – All Streaks
          </button>
        </div>

        {mode && (
          <form onSubmit={handleQ2Submit}>
            <div className="row mb-3">
              {mode === 1 ? (
                <>
                  <div className="col-md-4">
                    <input
                      type="number"
                      className="form-control"
                      placeholder="From (e.g. 2)"
                      value={startRange}
                      onChange={(e) => setStartRange(e.target.value)}
                      required
                    />
                  </div>
                  <div className="col-md-4">
                    <input
                      type="number"
                      className="form-control"
                      placeholder="To (e.g. 5)"
                      value={endRange}
                      onChange={(e) => setEndRange(e.target.value)}
                      required
                    />
                  </div>
                </>
              ) : (
                <div className="col-md-8">
                  <input
                    type="number"
                    className="form-control"
                    placeholder="Enter consecutive count (e.g. 10)"
                    value={targetCount}
                    onChange={(e) => setTargetCount(e.target.value)}
                    required
                  />
                </div>
              )}
              <div className="col-md-4">
                <button type="submit" className="btn btn-primary w-100">
                  Find
                </button>
              </div>
            </div>
          </form>
        )}

        {q2Loading && (
          <div className="text-center my-3">
            <div
              className="spinner-border text-primary"
              role="status"
              style={{ width: "3rem", height: "3rem" }}
            ></div>
            <p className="mt-2 text-secondary fw-semibold">
              Searching consecutive Harshad groups...
            </p>
          </div>
        )}

        {q2Result && !q2Loading && (
          <div className="mt-4 bg-light p-3 rounded">
            {mode === 1 && q2Result.range_results ? (
              <>
                <h5 className="text-success fw-semibold mb-3">
                  ✅ Groups within Range {startRange} – {endRange}
                </h5>
                <table className="table table-striped table-bordered">
                  <thead className="table-dark">
                    <tr>
                      <th>Group Size</th>
                      <th>Number of Groups</th>
                      <th>Groups Found</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(q2Result.range_results).map(
                      ([size, groups], idx) => {
                        let displayGroups = [];

                        if (Array.isArray(groups)) {
                          if (Array.isArray(groups[0])) displayGroups = groups;
                          else displayGroups = [groups];
                        } else if (typeof groups === "string") {
                          displayGroups = [[groups]];
                        } else if (typeof groups === "number") {
                          displayGroups = [[groups]];
                        }

                        return (
                          <tr key={idx}>
                            <td>{size}</td>
                            <td>{displayGroups.length}</td>
                            <td style={{ wordBreak: "break-all" }}>
                              {displayGroups
                                .map((g) =>
                                  Array.isArray(g)
                                    ? `[${g.join(", ")}]`
                                    : `[${g}]`
                                )
                                .join(" ‎ ‎ ‎ ")}
                            </td>
                          </tr>
                        );
                      }
                    )}
                  </tbody>
                </table>
              </>
            ) : mode === 2 && q2Result.streaks ? (
              <>
                <h5 className="text-success fw-semibold">
                  ✅ Found {q2Result.count} streaks of{" "}
                  {q2Result.streaks[0]?.length}-consecutive Harshads
                </h5>
                <ul className="list-group mt-2">
                  {q2Result.streaks.map((seq, idx) => (
                    <li
                      key={idx}
                      className="list-group-item text-dark fw-medium"
                    >
                      {seq.join(", ")}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="text-danger">❌ No results found.</p>
            )}
          </div>
        )}

        <div className="mt-4 p-3 bg-white rounded border border-secondary">
          <h6 className="fw-semibold text-primary">
            Why there are no 20 or more consecutive Harshad numbers?
          </h6>
          <p className="text-dark">
            Because the divisibility rule depends on each number’s digit sum.
            Eventually one number’s sum will not divide evenly, breaking the
            sequence.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Harshad;
