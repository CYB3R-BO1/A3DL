import axios from "axios";

export const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 60000,
});

export type AttackPayload = {
  attack_type: "fgsm" | "pgd";
  dataset: "cifar10" | "mnist";
  model_name: "simple_cnn";
  epsilon: number;
  alpha: number;
  steps: number;
  sample_limit: number;
  batch_size: number;
  checkpoint_path?: string | null;
};

export async function runAttack(payload: AttackPayload) {
  const { data } = await api.post("/attack/run", payload);
  return data;
}

export async function runDetect(runId: string) {
  const { data } = await api.post("/detect", { run_id: runId });
  return data;
}

export async function runDefend(runId: string, dataset: "cifar10" | "mnist") {
  const { data } = await api.post("/defend", { run_id: runId, dataset });
  return data;
}

export async function runReport(runId: string) {
  const { data } = await api.post("/report", { run_id: runId, include_detection: true, include_defense: true });
  return data;
}

export async function trainModel(payload: {
  dataset: "cifar10" | "mnist";
  epochs: number;
  batch_size: number;
  learning_rate: number;
  max_batches_per_epoch: number;
}) {
  const { data } = await api.post("/train", payload);
  return data;
}

export async function listExperiments(limit = 50) {
  const { data } = await api.get("/experiments", { params: { limit } });
  return data;
}
