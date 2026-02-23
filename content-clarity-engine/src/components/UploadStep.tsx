import { useState, useCallback, useRef } from "react";
import { Upload, X, FileText, FileType, File, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { UploadedFile, formatFileSize, getTotalSize } from "@/data/sampleData";
import { toast } from "sonner";

interface UploadStepProps {
  files: UploadedFile[];
  setFiles: (files: UploadedFile[]) => void;
  onContinue: () => void;
}

const FILE_ICONS: Record<string, React.ReactNode> = {
  PDF: <FileText className="w-5 h-5 text-destructive" />,
  DOCX: <FileType className="w-5 h-5 text-info" />,
  MD: <File className="w-5 h-5 text-foreground-muted" />,
  TXT: <File className="w-5 h-5 text-foreground-muted" />,
  HTML: <File className="w-5 h-5 text-warning" />,
};

const API_BASE = "http://localhost:5001";

const UploadStep = ({ files, setFiles, onContinue }: UploadStepProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadToBackend = useCallback(
    async (fileList: FileList | File[]) => {
      setIsUploading(true);
      try {
        const formData = new FormData();
        Array.from(fileList).forEach((f) => formData.append("files", f));

        const res = await fetch(`${API_BASE}/api/upload`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error(`Upload failed: ${res.status}`);

        const data = await res.json();
        const uploaded: UploadedFile[] = data.files.map((f: any) => ({
          id: f.id,
          name: f.name,
          size: f.size,
          type: f.type,
        }));

        setFiles([...files, ...uploaded]);
        toast.success(`${uploaded.length} file${uploaded.length !== 1 ? "s" : ""} uploaded`);
      } catch (err: any) {
        toast.error(err.message || "Failed to upload files. Is the backend running?");
      } finally {
        setIsUploading(false);
      }
    },
    [files, setFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files.length > 0) {
        uploadToBackend(e.dataTransfer.files);
      }
    },
    [uploadToBackend]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files || e.target.files.length === 0) return;
      uploadToBackend(e.target.files);
      e.target.value = "";
    },
    [uploadToBackend]
  );

  const removeFile = useCallback(
    (id: string) => setFiles(files.filter((f) => f.id !== id)),
    [files, setFiles]
  );

  const loadSamples = useCallback(async () => {
    setIsUploading(true);
    try {
      const res = await fetch(`${API_BASE}/api/load-samples`, { method: "POST" });
      if (!res.ok) throw new Error("Backend not available");
      const data = await res.json();
      const uploaded: UploadedFile[] = data.files.map((f: any) => ({
        id: f.id,
        name: f.name,
        size: f.size,
        type: f.type,
      }));
      setFiles(uploaded);
      toast.success(`${uploaded.length} sample documents loaded`);
    } catch {
      toast.error("Could not load sample documents. Make sure the backend is running (cd backend && python server.py)");
    } finally {
      setIsUploading(false);
    }
  }, [setFiles]);

  return (
    <div className="max-w-3xl mx-auto px-4">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Upload Your Documents</h1>
        <p className="text-foreground-muted max-w-xl mx-auto">
          Drop your document files below to begin scanning for contradictions, semantic drift, stale
          references, and terminology inconsistencies.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => !isUploading && inputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative min-h-[280px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors duration-200 ${isUploading
          ? "border-primary/40 bg-primary-light pointer-events-none"
          : isDragging
            ? "border-primary bg-primary-light"
            : "border-border-dashed bg-background-subtle hover:border-primary/40"
          }`}
      >
        {isUploading ? (
          <>
            <Loader2 className="w-12 h-12 text-primary animate-spin" />
            <p className="text-lg font-medium">Uploading & processing...</p>
            <p className="text-sm text-foreground-muted">Extracting text from your documents</p>
          </>
        ) : (
          <>
            <Upload className={`w-12 h-12 ${isDragging ? "text-primary" : "text-foreground-muted"}`} />
            <p className="text-lg font-medium">{isDragging ? "Drop files here" : "Drag & drop files here"}</p>
            <p className="text-sm text-foreground-muted">or click to browse</p>
            <div className="flex gap-2 mt-2">
              {["PDF", "DOCX", "HTML", "MD", "TXT"].map((fmt) => (
                <Badge key={fmt} variant="secondary" className="text-xs">
                  {fmt}
                </Badge>
              ))}
            </div>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.html,.md,.txt"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-6 border border-border rounded-lg overflow-hidden">
          {files.map((file) => (
            <div
              key={file.id}
              className="flex items-center justify-between px-4 py-3 border-b border-border last:border-b-0 hover:bg-background-subtle transition-colors"
            >
              <div className="flex items-center gap-3">
                {FILE_ICONS[file.type] || <File className="w-5 h-5 text-foreground-muted" />}
                <span className="text-sm font-medium">{file.name}</span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  {file.type}
                </Badge>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-foreground-muted">{formatFileSize(file.size)}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(file.id);
                  }}
                  className="text-destructive hover:text-destructive/80 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
          <div className="px-4 py-2.5 bg-background-subtle text-xs text-foreground-muted">
            {files.length} file{files.length !== 1 ? "s" : ""} selected · Total size: {getTotalSize(files)}
          </div>
        </div>
      )}

      {/* Sample docs */}
      <div className="mt-6">
        <Separator className="mb-4" />
        <div className="text-center">
          <button
            onClick={loadSamples}
            disabled={isUploading}
            className="text-sm text-primary font-medium hover:underline disabled:opacity-50"
          >
            Or try with sample documents →
          </button>
        </div>
      </div>

      {/* Continue */}
      <div className="flex justify-center mt-8">
        <Button
          size="lg"
          disabled={files.length === 0 || isUploading}
          onClick={onContinue}
          className="w-full max-w-md"
        >
          Continue to Configuration →
        </Button>
      </div>
    </div>
  );
};

export default UploadStep;

