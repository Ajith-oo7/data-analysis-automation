import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formatdate
from typing import List, Dict, Any, Optional
from pathlib import Path
import traceback

# Configure logging
logger = logging.getLogger(__name__)

class EmailSender:
    """
    Utility class for sending email notifications with report attachments.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email sender with configuration.
        
        Args:
            config: Configuration dictionary containing email settings
        """
        self.config = config
        self.email_config = config.get("reporting", {}).get("email", {})
        
        # Get email settings from environment variables or config
        self.smtp_server = os.environ.get("EMAIL_SMTP_SERVER", self.email_config.get("smtp_server", ""))
        self.smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", self.email_config.get("smtp_port", 587)))
        self.smtp_username = os.environ.get("EMAIL_USERNAME", self.email_config.get("username", ""))
        self.smtp_password = os.environ.get("EMAIL_PASSWORD", self.email_config.get("password", ""))
        self.sender_email = os.environ.get("EMAIL_SENDER", self.email_config.get("sender", ""))
        self.use_ssl = self.email_config.get("use_ssl", False)
        self.use_tls = self.email_config.get("use_tls", True)
        
        self.enabled = self.email_config.get("enabled", False) and self.smtp_server and self.smtp_username and self.smtp_password
        
        if not self.enabled:
            logger.warning("Email notifications are disabled or not properly configured")
        else:
            logger.info(f"Email sender initialized with server {self.smtp_server}:{self.smtp_port}")
    
    def send_report_notification(self, 
                                recipients: List[str], 
                                subject: str, 
                                message: str, 
                                report_paths: List[str],
                                data_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send email notification with report attachments.
        
        Args:
            recipients: List of email recipients
            subject: Email subject
            message: Email message body
            report_paths: List of paths to report files to attach
            data_summary: Optional data summary to include in the email
            
        Returns:
            Dictionary with status and message
        """
        if not self.enabled:
            return {
                "status": "error",
                "message": "Email notifications are disabled or not properly configured"
            }
        
        if not recipients:
            return {
                "status": "error",
                "message": "No recipients specified"
            }
        
        try:
            # Create message container
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(recipients)
            msg['Date'] = formatdate(localtime=True)
            msg['Subject'] = subject
            
            # Add data summary if provided
            if data_summary:
                summary_html = f"""
                <div style="margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                    <h3 style="margin-top: 0; color: #4a6fd4;">Data Summary</h3>
                    <p><strong>File:</strong> {data_summary.get('file_name', 'N/A')}</p>
                    <p><strong>Rows:</strong> {data_summary.get('num_rows', 'N/A')}</p>
                    <p><strong>Columns:</strong> {data_summary.get('num_columns', 'N/A')}</p>
                    <p><strong>Missing Values:</strong> {data_summary.get('missing_percentage', '0')}%</p>
                </div>
                """
                message = f"{message}\n\n{summary_html}"
            
            # Create HTML message
            html_message = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    h2 {{ color: #4a6fd4; }}
                    .footer {{ margin-top: 30px; font-size: 12px; color: #999; border-top: 1px solid #eee; padding-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>AI Data Analysis Report</h2>
                    {message}
                    <p>Please find the attached report(s) for your review.</p>
                    <div class="footer">
                        <p>This is an automated message from the AI Data Analysis Automation Platform.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Attach HTML message
            msg.attach(MIMEText(html_message, 'html'))
            
            # Attach reports
            for report_path in report_paths:
                if not os.path.exists(report_path):
                    logger.warning(f"Report file not found: {report_path}")
                    continue
                    
                file_name = os.path.basename(report_path)
                with open(report_path, "rb") as f:
                    attachment = MIMEApplication(f.read(), _subtype="html")
                    attachment.add_header('Content-Disposition', 'attachment', filename=file_name)
                    msg.attach(attachment)
            
            # Connect to server and send email
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                
                if self.use_tls:
                    server.starttls()
            
            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(self.sender_email, recipients, msg.as_string())
            server.close()
            
            logger.info(f"Email notification sent to {len(recipients)} recipients")
            return {
                "status": "success",
                "message": f"Email notification sent to {len(recipients)} recipients"
            }
            
        except Exception as e:
            error_msg = f"Error sending email notification: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": error_msg
            }

    def send_batch_summary(self, 
                          recipients: List[str], 
                          batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send batch processing summary email.
        
        Args:
            recipients: List of email recipients
            batch_results: Dictionary with batch processing results
            
        Returns:
            Dictionary with status and message
        """
        if not self.enabled:
            return {
                "status": "error",
                "message": "Email notifications are disabled or not properly configured"
            }
        
        try:
            # Extract batch info
            processed_files = batch_results.get("processed_files", [])
            successes = sum(1 for r in processed_files if r.get("status") == "success")
            failures = sum(1 for r in processed_files if r.get("status") == "error")
            
            # Create message
            subject = f"Batch Processing Summary: {successes} successes, {failures} failures"
            
            # Create HTML message with table of results
            results_table = """
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <tr style="background-color: #4a6fd4; color: white;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">File</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Status</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Details</th>
                </tr>
            """
            
            for result in processed_files:
                status = result.get("status", "")
                file_path = result.get("file_path", "")
                file_name = os.path.basename(file_path) if file_path else "Unknown"
                details = result.get("message", "")
                
                row_color = "#e6f7e6" if status == "success" else "#f7e6e6"
                status_text = "✅ Success" if status == "success" else "❌ Error"
                
                results_table += f"""
                <tr style="background-color: {row_color};">
                    <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{file_name}</td>
                    <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{status_text}</td>
                    <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{details}</td>
                </tr>
                """
            
            results_table += "</table>"
            
            message = f"""
            <p>A batch processing job has completed with the following results:</p>
            <ul>
                <li><strong>Total files:</strong> {len(processed_files)}</li>
                <li><strong>Successful:</strong> {successes}</li>
                <li><strong>Failed:</strong> {failures}</li>
            </ul>
            
            <h3 style="margin-top: 20px; color: #4a6fd4;">Processing Details</h3>
            {results_table}
            """
            
            # Send email
            return self.send_report_notification(
                recipients=recipients,
                subject=subject,
                message=message,
                report_paths=[]  # No attachments for batch summary
            )
            
        except Exception as e:
            error_msg = f"Error sending batch summary email: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": error_msg
            }

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    config = {
        "reporting": {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "your_username",
                "password": "your_password",
                "sender": "reports@example.com",
                "use_tls": True
            }
        }
    }
    
    # Initialize email sender
    email_sender = EmailSender(config)
    
    # Example: Send a report notification
    result = email_sender.send_report_notification(
        recipients=["user@example.com"],
        subject="Data Analysis Report",
        message="Here is your data analysis report.",
        report_paths=["path/to/report.html"],
        data_summary={
            "file_name": "sample_data.csv",
            "num_rows": 1000,
            "num_columns": 15,
            "missing_percentage": 2.5
        }
    )
    
    print(f"Email send result: {result['status']}") 