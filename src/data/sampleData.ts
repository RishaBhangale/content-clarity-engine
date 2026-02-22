export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
}

export interface ScanConfig {
  detectionTypes: {
    contradiction: boolean;
    semanticDrift: boolean;
    staleReference: boolean;
    terminology: boolean;
  };
  sensitivity: "high" | "medium" | "low";
  scope: "cross" | "within" | "both";
}

export type FindingSeverity = "critical" | "warning" | "info";
export type FindingType = "Contradiction" | "Semantic Drift" | "Stale Reference" | "Terminology";

export interface Finding {
  id: number;
  severity: FindingSeverity;
  title: string;
  sourceA: string;
  sourceB?: string;
  type: FindingType;
  excerptA: string;
  excerptB?: string;
  suggestion: string;
  confidence: number;
}

export const SAMPLE_FILES: UploadedFile[] = [
  { id: "1", name: "Data-Retention-Policy-v3.pdf", size: 245000, type: "PDF" },
  { id: "2", name: "SOP-Data-Handling-Procedures.docx", size: 189000, type: "DOCX" },
  { id: "3", name: "Employee-Onboarding-Guide-2024.pdf", size: 312000, type: "PDF" },
  { id: "4", name: "HR-Manual-Section7-PTO.docx", size: 156000, type: "DOCX" },
  { id: "5", name: "Deploy-Guide-v3.md", size: 45000, type: "MD" },
  { id: "6", name: "Security-Controls-Audit-2025.pdf", size: 478000, type: "PDF" },
];

export const SAMPLE_FINDINGS: Finding[] = [
  {
    id: 1,
    severity: "critical",
    title: "Data retention period conflict",
    sourceA: "Data-Retention-Policy-v3.pdf §3.2",
    sourceB: "SOP-Data-Handling-Procedures.docx §1.4",
    type: "Contradiction",
    excerptA: '"All customer data must be retained for a minimum of 90 days after account closure, after which it shall be permanently deleted."',
    excerptB: '"Customer data shall be retained for a minimum of 1 year following account termination to comply with regulatory requirements."',
    suggestion: "Align both documents to the longer retention period (1 year) if regulatory requirements mandate it, and update the Data Retention Policy accordingly.",
    confidence: 97,
  },
  {
    id: 2,
    severity: "critical",
    title: "PTO accrual method differs",
    sourceA: "HR-Manual-Section7-PTO.docx §2.1",
    sourceB: "Employee-Onboarding-Guide-2024.pdf §5.3",
    type: "Contradiction",
    excerptA: '"Paid time off is accrued on a monthly basis at a rate of 1.25 days per month of continuous employment."',
    excerptB: '"New employees accrue PTO on a bi-weekly basis starting from their first pay period."',
    suggestion: "Confirm with HR which accrual method is current policy and update the outdated document to match.",
    confidence: 95,
  },
  {
    id: 3,
    severity: "critical",
    title: "Security review frequency mismatch",
    sourceA: "Security-Controls-Audit-2025.pdf §4.1",
    sourceB: "SOP-Data-Handling-Procedures.docx §6.2",
    type: "Contradiction",
    excerptA: '"Security control reviews must be conducted on a quarterly basis, with findings reported to the CISO within 5 business days."',
    excerptB: '"Annual security reviews are conducted by the compliance team and results shared during the yearly audit meeting."',
    suggestion: "Adopt the quarterly review schedule as it provides stronger security posture. Update the SOP to reflect quarterly cadence.",
    confidence: 93,
  },
  {
    id: 4,
    severity: "warning",
    title: "Deployment approval process diverged",
    sourceA: "Deploy-Guide-v3.md §2",
    sourceB: "SOP-Data-Handling-Procedures.docx §8.1",
    type: "Semantic Drift",
    excerptA: '"Submit deployment requests via the Slack #deploy-requests channel. If no objection is raised within 24 hours, the deployment is auto-approved."',
    excerptB: '"All deployment requests must be submitted through the ServiceDesk portal. Manager approval is required within 48 business hours before proceeding."',
    suggestion: "Determine the authoritative deployment process and update both documents. Consider that the Slack-based process may lack proper audit trails.",
    confidence: 88,
  },
  {
    id: 5,
    severity: "warning",
    title: "Incident response escalation paths differ",
    sourceA: "Security-Controls-Audit-2025.pdf §7.3",
    sourceB: "SOP-Data-Handling-Procedures.docx §9.1",
    type: "Semantic Drift",
    excerptA: '"Critical incidents escalate to: Team Lead → Security Manager → CISO → CTO within 1 hour."',
    excerptB: '"Incidents are reported to the department manager who decides whether to escalate to the IT Security team."',
    suggestion: "Standardize the escalation chain. The Security Audit document's explicit chain is preferable for audit compliance.",
    confidence: 85,
  },
  {
    id: 6,
    severity: "warning",
    title: "Onboarding checklist steps diverged",
    sourceA: "Employee-Onboarding-Guide-2024.pdf §3",
    sourceB: "HR-Manual-Section7-PTO.docx §1.2",
    type: "Semantic Drift",
    excerptA: '"The onboarding process consists of 12 mandatory steps, including IT setup, compliance training, and team introductions."',
    excerptB: '"New hire onboarding follows 8 key steps as outlined in the standard HR process."',
    suggestion: "Reconcile the two checklists and create a single authoritative onboarding process document.",
    confidence: 82,
  },
  {
    id: 7,
    severity: "warning",
    title: "Data classification levels inconsistent",
    sourceA: "Data-Retention-Policy-v3.pdf §2.1",
    sourceB: "Security-Controls-Audit-2025.pdf §3.2",
    type: "Contradiction",
    excerptA: '"Data is classified into three levels: Public, Internal, and Confidential."',
    excerptB: '"The organization uses four data classification levels: Public, Internal, Confidential, and Restricted."',
    suggestion: "Adopt the four-level classification system from the Security Audit and update the Retention Policy to include 'Restricted' level.",
    confidence: 79,
  },
  {
    id: 8,
    severity: "warning",
    title: "Password policy requirements differ",
    sourceA: "Security-Controls-Audit-2025.pdf §5.1",
    sourceB: "Employee-Onboarding-Guide-2024.pdf §8.2",
    type: "Contradiction",
    excerptA: '"Passwords must be a minimum of 12 characters and include uppercase, lowercase, numbers, and special characters."',
    excerptB: '"Set your password with at least 8 characters. We recommend including a mix of letters and numbers."',
    suggestion: "Update the Onboarding Guide to reflect the stricter 12-character requirement from the Security Audit.",
    confidence: 91,
  },
  {
    id: 9,
    severity: "info",
    title: "Deploy Guide references Jenkins CI (deprecated)",
    sourceA: "Deploy-Guide-v3.md §4.1",
    type: "Stale Reference",
    excerptA: '"Continuous integration pipelines run on Jenkins CI. Configure your Jenkinsfile in the repository root."',
    excerptB: '"CI/CD pipelines have been migrated to GitHub Actions as of Q3 2025." — Security-Controls-Audit-2025.pdf §10.2',
    suggestion: "Update Deploy-Guide-v3.md to reference GitHub Actions instead of Jenkins CI. Include updated pipeline configuration examples.",
    confidence: 96,
  },
  {
    id: 10,
    severity: "info",
    title: "Terminology: 'churn rate' vs 'customer attrition' vs 'logo loss'",
    sourceA: "Data-Retention-Policy-v3.pdf, SOP-Data-Handling-Procedures.docx, Employee-Onboarding-Guide-2024.pdf",
    type: "Terminology",
    excerptA: '"Data-Retention-Policy uses \"churn rate\", SOP uses \"customer attrition\", and Employee-Onboarding-Guide uses \"logo loss\" — all referring to the same metric."',
    suggestion: "Standardize on \"Customer Churn Rate\" as the canonical term across all documents. Add to the corporate glossary.",
    confidence: 90,
  },
  {
    id: 11,
    severity: "info",
    title: "Terminology: 'offboarding' vs 'exit process' vs 'separation procedure'",
    sourceA: "HR-Manual-Section7-PTO.docx, Employee-Onboarding-Guide-2024.pdf, SOP-Data-Handling-Procedures.docx",
    type: "Terminology",
    excerptA: '"Three documents use different terms — \"offboarding\" (HR Manual), \"exit process\" (Onboarding Guide), and \"separation procedure\" (SOP) — for the same workflow."',
    suggestion: "Standardize on \"Offboarding\" as it is the most widely understood term. Update all documents and add to glossary.",
    confidence: 87,
  },
  {
    id: 12,
    severity: "info",
    title: "Onboarding Guide references SharePoint 2019 (migrated to SharePoint Online)",
    sourceA: "Employee-Onboarding-Guide-2024.pdf §6.1",
    type: "Stale Reference",
    excerptA: '"Upload your completed forms to the SharePoint 2019 HR portal at https://intranet.company.com/hr/forms."',
    suggestion: "Update the reference to SharePoint Online. The organization migrated in January 2025. Update URLs and access instructions.",
    confidence: 94,
  },
];

export const DEFAULT_SCAN_CONFIG: ScanConfig = {
  detectionTypes: {
    contradiction: true,
    semanticDrift: true,
    staleReference: true,
    terminology: true,
  },
  sensitivity: "medium",
  scope: "cross",
};

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

export function getTotalSize(files: UploadedFile[]): string {
  const total = files.reduce((sum, f) => sum + f.size, 0);
  return formatFileSize(total);
}
