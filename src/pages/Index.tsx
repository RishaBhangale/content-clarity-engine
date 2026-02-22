import { useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import StepIndicator from "@/components/StepIndicator";
import UploadStep from "@/components/UploadStep";
import ConfigureStep from "@/components/ConfigureStep";
import ScanningStep from "@/components/ScanningStep";
import ResultsStep from "@/components/ResultsStep";
import { UploadedFile, ScanConfig, DEFAULT_SCAN_CONFIG } from "@/data/sampleData";

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

const Index = () => {
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [config, setConfig] = useState<ScanConfig>(DEFAULT_SCAN_CONFIG);

  const reset = useCallback(() => {
    setStep(1);
    setFiles([]);
    setConfig(DEFAULT_SCAN_CONFIG);
  }, []);

  return (
    <div className="min-h-screen bg-background-subtle">
      <Navbar onReset={reset} />
      <StepIndicator currentStep={step} />

      <main className="pb-16">
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div key="upload" variants={pageVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.25 }}>
              <UploadStep files={files} setFiles={setFiles} onContinue={() => setStep(2)} />
            </motion.div>
          )}
          {step === 2 && (
            <motion.div key="configure" variants={pageVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.25 }}>
              <ConfigureStep files={files} config={config} setConfig={setConfig} onBack={() => setStep(1)} onStart={() => setStep(3)} />
            </motion.div>
          )}
          {step === 3 && (
            <motion.div key="scanning" variants={pageVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.25 }}>
              <ScanningStep fileCount={files.length} onComplete={() => setStep(4)} />
            </motion.div>
          )}
          {step === 4 && (
            <motion.div key="results" variants={pageVariants} initial="initial" animate="animate" exit="exit" transition={{ duration: 0.25 }}>
              <ResultsStep files={files} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default Index;
