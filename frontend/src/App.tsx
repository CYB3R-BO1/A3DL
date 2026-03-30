import { ChangeEvent, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { listExperiments, runAttack, runDefend, runDetect, runReport, trainModel } from "./services/api";

type AttackData = {
  run_id: string;
  metrics: {
    clean_accuracy: number;
    adversarial_accuracy: number;
    attack_success_rate: number;
    avg_confidence_drop: number;
  };
  samples: Array<{
    sample_index: number;
    true_label: number;
    original_image_path: string;
    adversarial_image_path: string;
    perturbation_image_path: string;
  }>;
};

export function App() {
  const [dataset, setDataset] = useState<"cifar10" | "mnist">("cifar10");
  const [attackType, setAttackType] = useState<"fgsm" | "pgd">("fgsm");
  const [epsilon, setEpsilon] = useState(8 / 255);
  const [steps, setSteps] = useState(10);
  const [loading, setLoading] = useState(false);
  const [attackData, setAttackData] = useState<AttackData | null>(null);
  const [detectData, setDetectData] = useState<any>(null);
  const [defendData, setDefendData] = useState<any>(null);
  const [reportData, setReportData] = useState<any>(null);
  const [trainData, setTrainData] = useState<any>(null);
  const [checkpointPath, setCheckpointPath] = useState("");
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const chartData = useMemo(() => {
    if (!attackData) {
      return [];
    }
    return [
      { name: "Clean Acc", value: attackData.metrics.clean_accuracy },
      { name: "Adv Acc", value: attackData.metrics.adversarial_accuracy },
      { name: "Attack SR", value: attackData.metrics.attack_success_rate },
    ];
  }, [attackData]);

  async function executeAttack() {
    setLoading(true);
    setError(null);
    try {
      const data = await runAttack({
        attack_type: attackType,
        dataset,
        model_name: "simple_cnn",
        epsilon,
        alpha: 2 / 255,
        steps,
        sample_limit: 64,
        batch_size: 32,
        checkpoint_path: checkpointPath || null,
      });
      setAttackData(data);
      setDetectData(null);
      setDefendData(null);
      setReportData(null);
    } catch (e: any) {
      setError(e.message ?? "Failed to run attack");
    } finally {
      setLoading(false);
    }
  }

  async function executeTrain() {
    setLoading(true);
    setError(null);
    try {
      const data = await trainModel({
        dataset,
        epochs: 1,
        batch_size: 64,
        learning_rate: 1e-3,
        max_batches_per_epoch: 100,
      });
      setTrainData(data);
      setCheckpointPath(data.checkpoint_path);
    } catch (e: any) {
      setError(e.message ?? "Failed to train model");
    } finally {
      setLoading(false);
    }
  }

  async function loadHistory() {
    setLoading(true);
    setError(null);
    try {
      const data = await listExperiments(25);
      setHistoryData(data.experiments ?? []);
    } catch (e: any) {
      setError(e.message ?? "Failed to load experiment history");
    } finally {
      setLoading(false);
    }
  }

  async function executePipeline() {
    if (!attackData) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const detect = await runDetect(attackData.run_id);
      setDetectData(detect);
      const defend = await runDefend(attackData.run_id, dataset);
      setDefendData(defend);
      const report = await runReport(attackData.run_id);
      setReportData(report);
    } catch (e: any) {
      setError(e.message ?? "Failed to run pipeline");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-sand text-ink">
      <header className="border-b border-ink/20 bg-gradient-to-r from-mint to-sand px-6 py-4">
        <h1 className="text-2xl font-semibold">A3DL: Autonomous Adversarial Attack and Defense Lab</h1>
        <p className="text-sm">AI red-team MVP dashboard for attack, detection, defense, and reporting.</p>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-3">
        <section className="rounded-lg bg-white p-4 shadow-sm lg:col-span-1">
          <h2 className="mb-3 text-lg font-semibold">Attack Config</h2>
          <label className="mb-2 block text-sm">Dataset</label>
          <select
            className="mb-3 w-full rounded border p-2"
            value={dataset}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setDataset(e.target.value as "cifar10" | "mnist")}
          >
            <option value="cifar10">CIFAR-10</option>
            <option value="mnist">MNIST</option>
          </select>

          <label className="mb-2 block text-sm">Attack Type</label>
          <select
            className="mb-3 w-full rounded border p-2"
            value={attackType}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setAttackType(e.target.value as "fgsm" | "pgd")}
          >
            <option value="fgsm">FGSM</option>
            <option value="pgd">PGD</option>
          </select>

          <label className="mb-2 block text-sm">Epsilon: {epsilon.toFixed(4)}</label>
          <input
            className="mb-3 w-full"
            type="range"
            min={1 / 255}
            max={32 / 255}
            step={1 / 255}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
          />

          <label className="mb-2 block text-sm">PGD Steps</label>
          <input
            className="mb-4 w-full rounded border p-2"
            type="number"
            min={1}
            max={200}
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />

          <label className="mb-2 block text-sm">Checkpoint Path (optional)</label>
          <input
            className="mb-4 w-full rounded border p-2"
            type="text"
            value={checkpointPath}
            onChange={(e) => setCheckpointPath(e.target.value)}
            placeholder="../artifacts/models/cifar10_simple_cnn_*.pt"
          />

          <button className="mb-2 w-full rounded bg-mint px-3 py-2 text-ink disabled:opacity-40" onClick={executeTrain} disabled={loading}>
            Train + Save Checkpoint
          </button>

          <button className="mb-2 w-full rounded bg-ink px-3 py-2 text-white disabled:opacity-40" onClick={executeAttack} disabled={loading}>
            Run Attack
          </button>
          <button className="w-full rounded bg-ember px-3 py-2 text-white disabled:opacity-40" onClick={executePipeline} disabled={loading || !attackData}>
            Run Detect + Defend + Report
          </button>
          <button className="mt-2 w-full rounded border border-ink px-3 py-2 text-ink disabled:opacity-40" onClick={loadHistory} disabled={loading}>
            Load Experiment History
          </button>
          {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
        </section>

        <section className="rounded-lg bg-white p-4 shadow-sm lg:col-span-2">
          <h2 className="mb-3 text-lg font-semibold">Metrics</h2>
          {attackData ? (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <MetricCard label="Clean Acc" value={attackData.metrics.clean_accuracy} />
                <MetricCard label="Adv Acc" value={attackData.metrics.adversarial_accuracy} />
                <MetricCard label="Attack Success" value={attackData.metrics.attack_success_rate} />
                <MetricCard label="Conf Drop" value={attackData.metrics.avg_confidence_drop} />
              </div>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#12263a" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-600">Run an attack to view metrics and visual artifacts.</p>
          )}

          {attackData && attackData.samples.length > 0 && (
            <div className="mt-6">
              <h3 className="mb-2 text-md font-semibold">Sample Artifacts (paths)</h3>
              <ul className="max-h-44 overflow-auto rounded border p-2 text-xs">
                {attackData.samples.slice(0, 10).map((s) => (
                  <li key={s.sample_index} className="mb-2 border-b pb-1">
                    #{s.sample_index} | orig: {s.original_image_path} | adv: {s.adversarial_image_path} | pert: {s.perturbation_image_path}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <section className="rounded-lg bg-white p-4 shadow-sm lg:col-span-3">
          <h2 className="mb-3 text-lg font-semibold">Training Output</h2>
          <JsonCard title="Train" data={trainData} />
        </section>

        <section className="rounded-lg bg-white p-4 shadow-sm lg:col-span-3">
          <h2 className="mb-3 text-lg font-semibold">Detection / Defense / Report</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <JsonCard title="Detection" data={detectData} />
            <JsonCard title="Defense" data={defendData} />
            <JsonCard title="Report" data={reportData} />
          </div>
        </section>

        <section className="rounded-lg bg-white p-4 shadow-sm lg:col-span-3">
          <h2 className="mb-3 text-lg font-semibold">Experiment History</h2>
          <pre className="max-h-80 overflow-auto rounded bg-gray-50 p-2 text-xs">
            {historyData.length > 0 ? JSON.stringify(historyData, null, 2) : "No history loaded yet."}
          </pre>
        </section>
      </main>
    </div>
  );
}

function MetricCard(props: { label: string; value: number }) {
  return (
    <div className="rounded border bg-sand p-3">
      <p className="text-xs uppercase tracking-wide">{props.label}</p>
      <p className="text-lg font-semibold">{props.value.toFixed(3)}</p>
    </div>
  );
}

function JsonCard(props: { title: string; data: any }) {
  return (
    <div className="rounded border p-3">
      <h3 className="mb-2 font-semibold">{props.title}</h3>
      <pre className="max-h-72 overflow-auto rounded bg-gray-50 p-2 text-xs">
        {props.data ? JSON.stringify(props.data, null, 2) : "No data yet."}
      </pre>
    </div>
  );
}
