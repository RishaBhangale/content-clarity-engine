import { useEffect, useState, useRef } from "react";
import { Progress } from "@/components/ui/progress";
import { Check, Loader2 } from "lucide-react";

interface ScanningStepProps {
  fileCount: number;
  onComplete: () => void;
}

const SCAN_STEPS = [
  { label: "Parsing documents and extracting text...", trigger: 0 },
  { label: "Building semantic embeddings...", trigger: 20 },
  { label: "Running pairwise contradiction analysis...", trigger: 40 },
  { label: "Detecting semantic drift patterns...", trigger: 60 },
  { label: "Scanning for stale references...", trigger: 75 },
  { label: "Mapping terminology clusters...", trigger: 90 },
  { label: "Generating findings report...", trigger: 95 },
];

const ScanningStep = ({ fileCount, onComplete }: ScanningStepProps) => {
  const [progress, setProgress] = useState(0);
  const [pairsCompared, setPairsCompared] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const totalPairs = (fileCount * (fileCount - 1)) / 2;
  const completedRef = useRef(false);

  useEffect(() => {
    const duration = fileCount * 5 * 1000; // total ms
    const interval = 50;
    const increment = (100 / duration) * interval;
    let current = 0;

    const timer = setInterval(() => {
      current = Math.min(current + increment, 100);
      setProgress(current);
      setPairsCompared(Math.min(Math.floor((current / 100) * totalPairs), totalPairs));

      if (current >= 100) {
        clearInterval(timer);
        setIsComplete(true);
        if (!completedRef.current) {
          completedRef.current = true;
          setTimeout(onComplete, 1200);
        }
      }
    }, interval);

    return () => clearInterval(timer);
  }, [fileCount, totalPairs, onComplete]);

  return (
    <div className="max-w-2xl mx-auto px-4 text-center">
      <h2 className="text-2xl font-bold mb-2">
        {isComplete ? "Scan Complete!" : "Analyzing Your Documents..."}
      </h2>
      <p className="text-foreground-muted mb-8">
        {isComplete
          ? "All documents have been analyzed. Preparing your results."
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
          const isDone = progress >= step.trigger + 10;
          const isActive = progress >= step.trigger && !isDone;

          return (
            <div
              key={step.label}
              className={`flex items-center gap-3 transition-opacity duration-300 ${
                progress >= step.trigger ? "opacity-100" : "opacity-30"
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
