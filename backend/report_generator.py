import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = self.styles['Title']
        self.heading_style = self.styles['Heading2']
        self.normal_style = self.styles['Normal']

    def generate_report(self, optimized_data, equal_weighted_data, historical_data, params):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # --- Page 1: Parameters & Allocation ---
        elements.append(Paragraph("Portfolio Optimization Report", self.title_style))
        elements.append(Spacer(1, 12))
        
        # Parameters
        params_data = [
            ["Parameter", "Value"],
            ["Start Date", params.get("start_date")],
            ["End Date", params.get("end_date")],
            ["Strategy", params.get("strategy")],
            ["Target Volatility", f"{params.get('target_volatility', 0):.2%}" if params.get('target_volatility') else "N/A"],
        ]
        t_params = Table(params_data, colWidths=[2*inch, 3*inch])
        t_params.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t_params)
        elements.append(Spacer(1, 24))

        # Allocation Table
        elements.append(Paragraph("Portfolio Allocation", self.heading_style))
        alloc_data = [["Ticker", "Weight"]]
        for ticker, weight in optimized_data["weights"].items():
            alloc_data.append([ticker, f"{weight:.2%}"])
            
        t_alloc = Table(alloc_data, colWidths=[2*inch, 2*inch])
        t_alloc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t_alloc)
        elements.append(PageBreak())

        # --- Page 2: Optimized Pie Chart ---
        elements.append(Paragraph("Optimized Portfolio Allocation", self.heading_style))
        pie_opt = self._create_pie_chart(optimized_data["weights"], "Optimized Portfolio")
        elements.append(Image(pie_opt, width=6*inch, height=4.5*inch))
        elements.append(PageBreak())

        # --- Page 3: Equal Weighted Pie Chart ---
        elements.append(Paragraph("Equal Weighted Portfolio Allocation", self.heading_style))
        pie_eq = self._create_pie_chart(equal_weighted_data["weights"], "Equal Weighted Portfolio")
        elements.append(Image(pie_eq, width=6*inch, height=4.5*inch))
        elements.append(PageBreak())

        # --- Page 4: Performance Metrics Comparison ---
        elements.append(Paragraph("Performance Metrics Comparison", self.heading_style))
        
        metrics_opt = optimized_data.get("metrics", {})
        metrics_eq = equal_weighted_data.get("metrics", {})
        
        # Filter key metrics
        keys = ["Annualized Return (CAGR)", "Expected Return", "Standard Deviation", "Best Year", "Worst Year", "Maximum Drawdown", "Sharpe Ratio", "Sortino Ratio"]
        
        comp_data = [["Metric", "Optimized", "Equal Weighted"]]
        for key in keys:
            val_opt = metrics_opt.get(key, 0)
            val_eq = metrics_eq.get(key, 0)
            
            # Format
            if "Ratio" in key:
                fmt = "{:.2f}"
            else:
                fmt = "{:.2%}"
                
            comp_data.append([key, fmt.format(val_opt), fmt.format(val_eq)])
            
        t_comp = Table(comp_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        t_comp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
        ]))
        elements.append(t_comp)
        elements.append(PageBreak())

        # --- Page 5: Growth Chart ---
        elements.append(Paragraph("Portfolio Growth (Cumulative Returns)", self.heading_style))
        growth_chart = self._create_growth_chart(optimized_data["weights"], equal_weighted_data["weights"], historical_data)
        elements.append(Image(growth_chart, width=6*inch, height=4.5*inch))
        elements.append(PageBreak())

        # --- Page 6: Annual Returns Bar Chart ---
        elements.append(Paragraph("Annual Returns", self.heading_style))
        bar_chart = self._create_bar_chart(optimized_data["weights"], equal_weighted_data["weights"], historical_data)
        elements.append(Image(bar_chart, width=6*inch, height=4.5*inch))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

    def _create_pie_chart(self, weights, title):
        plt.figure(figsize=(6, 4.5))
        labels = list(weights.keys())
        sizes = list(weights.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        img_buf.seek(0)
        return img_buf

    def _create_growth_chart(self, w_opt, w_eq, data):
        plt.figure(figsize=(8, 6))
        rets = data.pct_change().dropna()
        
        # Calculate Cumulative Returns
        port_ret_opt = (rets * pd.Series(w_opt)).sum(axis=1)
        cum_ret_opt = (1 + port_ret_opt).cumprod() * 10000
        
        port_ret_eq = (rets * pd.Series(w_eq)).sum(axis=1)
        cum_ret_eq = (1 + port_ret_eq).cumprod() * 10000
        
        plt.plot(cum_ret_opt.index, cum_ret_opt, label='Optimized', color='blue')
        plt.plot(cum_ret_eq.index, cum_ret_eq, label='Equal Weighted', color='black')
        plt.title("Portfolio Growth ($10,000 Start)")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        img_buf.seek(0)
        return img_buf

    def _create_bar_chart(self, w_opt, w_eq, data):
        plt.figure(figsize=(8, 6))
        rets = data.pct_change().dropna()
        
        port_ret_opt = (rets * pd.Series(w_opt)).sum(axis=1)
        port_ret_eq = (rets * pd.Series(w_eq)).sum(axis=1)
        
        annual_ret_opt = port_ret_opt.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        annual_ret_eq = port_ret_eq.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        years = annual_ret_opt.index.year
        x = np.arange(len(years))
        width = 0.35
        
        plt.bar(x - width/2, annual_ret_opt, width, label='Optimized', color='blue')
        plt.bar(x + width/2, annual_ret_eq, width, label='Equal Weighted', color='black')
        
        plt.ylabel('Return')
        plt.title('Annual Returns')
        plt.xticks(x, years, rotation=45)
        plt.legend()
        plt.grid(True, axis='y')
        plt.ylim(-0.3, 0.3) # As requested -30% to 30% (approx, or auto)
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        img_buf.seek(0)
        return img_buf
