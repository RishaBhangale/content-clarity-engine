import { useEffect, useState, useRef } from "react";
import { Progress } from "@/components/ui/progress";
import { Check, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadedFile, ScanConfig, Finding } from "@/data/sampleData";

interface ScanningStepProps {
  files: UploadedFile[];
  config: ScanConfig;
  onComplete: (findings: Finding[]) => void;
}

const SCAN_STEPS = [
  { label: "Parsing documents and extracting text...", trigger: 0 },
  { label: "Building semantic embeddings...", trigger: 15 },
  { label: "Running pairwise contradiction analysis...", trigger: 35 },
  { label: "Scoring contradiction candidates (NLI)...", trigger: 55 },
  { label: "Assembling findings report...", trigger: 80 },
];

const ScanningStep = ({ files, config, onComplete }: ScanningStepProps) => {
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [findings, setFindings] = useState<Finding[] | null>(null);
  const completedRef = useRef(false);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Start progress animation (simulated progress while waiting for backend)
    let current = 0;
    progressTimerRef.current = setInterval(() => {
      // Progress moves quickly at first, slows down as it approaches 85%
      // It sits at ~85% until the backend responds
      if (current < 85) {
        const increment = (85 - current) * 0.03;
        current = Math.min(current + Math.max(increment, 0.2), 85);
        setProgress(current);
      }
    }, 100);

    // Call the backend scan API
    const docIds = files.map(f => f.id);
    fetch("http://localhost:5001/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        docIds,
        config: {
          sensitivity: config.sensitivity,
          scope: config.scope,
          detectionTypes: config.detectionTypes,
        },
      }),
    })
      .then(res => {
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        return res.json();
      })
      .then(data => {
        // Backend responded — finish progress
        if (progressTimerRef.current) clearInterval(progressTimerRef.current);

        // Animate from current to 100%
        const finishInterval = setInterval(() => {
          current = Math.min(current + 3, 100);
          setProgress(current);
          if (current >= 100) {
            clearInterval(finishInterval);
            setIsComplete(true);
            setFindings(data.findings || []);
          }
        }, 30);
      })
      .catch(err => {
        if (progressTimerRef.current) clearInterval(progressTimerRef.current);
        setError(err.message || "Failed to connect to backend");
      });

    return () => {
      if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    };
  }, [files, config]);

  // Auto-proceed after completion
  useEffect(() => {
    if (isComplete && findings !== null && !completedRef.current) {
      completedRef.current = true;
      setTimeout(() => onComplete(findings), 1200);
    }
  }, [isComplete, findings, onComplete]);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto px-4 text-center">
        <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Scan Failed</h2>
        <p className="text-foreground-muted mb-4">
          Could not connect to the analysis backend.
        </p>
        <p className="text-sm text-destructive bg-destructive-light rounded-lg p-3 mb-6">
          {error}
        </p>
        <p className="text-sm text-foreground-muted">
          Make sure the Flask backend is running:{" "}
          <code className="bg-muted px-2 py-1 rounded text-xs">
            cd backend && python server.py
          </code>
        </p>
      </div>
    );
  }

  const fileCount = files.length;
  const totalPairs = (fileCount * (fileCount - 1)) / 2;
  const pairsCompared = Math.min(Math.floor((progress / 100) * totalPairs), totalPairs);

  return (
    <div className="max-w-2xl mx-auto px-4 text-center">
      <h2 className="text-2xl font-bold mb-2">
        {isComplete ? "Scan Complete!" : "Analyzing Your Documents..."}
      </h2>
      <p className="text-foreground-muted mb-8">
        {isComplete
          ? `Found ${findings?.length || 0} contradiction${(findings?.length || 0) !== 1 ? "s" : ""}. Preparing your results.`
          : `C1 is scanning ${fileCount} documents for inconsistencies.`}
      </p>

      <div className="mb-3">
        <Progress
          value={progress}
          className={`h-3 transition-colors ${isComplete ? "[&>div]:bg-secondary" : ""}`}
        />
      </div>
      <p className="text-sm text-foreground-muted mb-8">
        {Math.round(progress)}% · Compared {pairsCompared}/{totalPairs} document pairs
      </p>

      <div className="text-left space-y-3 bg-background-subtle rounded-lg p-5 border border-border">
        {SCAN_STEPS.map((step) => {
          const isDone = progress >= step.trigger + 15;
          const isActive = progress >= step.trigger && !isDone;

          return (
            <div
              key={step.label}
              className={`flex items-center gap-3 transition-opacity duration-300 ${progress >= step.trigger ? "opacity-100" : "opacity-30"
                }`}
            >
              {isDone ? (
                <Check className="w-4 h-4 text-secondary flex-shrink-0" />
              ) : isActive ? (
                <Loader2 className="w-4 h-4 text-primary flex-shrink-0 animate-spin" />
              ) : (
                <div className="w-4 h-4 rounded-full border border-border flex-shrink-0" />
              )}
              <span className={`text-sm ${isDone ? "text-foreground" : "text-foreground-muted"}`}>
                {isDone ? "✓ " : ""}
                {step.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ScanningStep;

