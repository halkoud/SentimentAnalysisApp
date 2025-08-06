import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import threading
import re

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis App - Group Project")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize NLTK components
        self.setup_nltk()
        
        # Create the UI
        self.create_widgets()
        
    def setup_nltk(self):
        """Download required NLTK data if not already present"""
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.sia = SentimentIntensityAnalyzer()
        
    def create_widgets(self):
        """Create and layout all UI widgets"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎭 Sentiment Analysis App", 
            font=('Arial', 20, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Team info
        team_label = tk.Label(
            self.root,
            text="Team: Huda (NLP) • Christian (UI) • Brayan + Asim (Testing & Presentation)",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        team_label.pack(pady=5)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Input label
        input_label = tk.Label(
            input_frame,
            text="Enter text to analyze:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#34495e'
        )
        input_label.pack(anchor='w', pady=(0, 5))
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=6,
            width=60,
            font=('Arial', 11),
            wrap=tk.WORD,
            relief='solid',
            borderwidth=1
        )
        self.text_input.pack(fill='both', expand=True, pady=(0, 10))
        
        # Analyze button
        self.analyze_btn = tk.Button(
            input_frame,
            text="🔍 Analyze Sentiment",
            command=self.analyze_sentiment,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.analyze_btn.pack(pady=10)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(pady=20, padx=20, fill='x')
        
        # Results label
        results_label = tk.Label(
            results_frame,
            text="Analysis Results:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#34495e'
        )
        results_label.pack(anchor='w', pady=(0, 10))
        
        # Sentiment result
        self.sentiment_result = tk.Label(
            results_frame,
            text="Enter text above and click 'Analyze Sentiment'",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#7f8c8d',
            relief='solid',
            borderwidth=1,
            pady=15,
            wraplength=500
        )
        self.sentiment_result.pack(fill='x', pady=(0, 10))
        
        # Detailed scores frame
        self.scores_frame = tk.Frame(results_frame, bg='#f0f0f0')
        self.scores_frame.pack(fill='x')
        
        # Progress bars for sentiment scores
        self.create_progress_bars()
        
        # Sample texts button
        sample_btn = tk.Button(
            self.root,
            text="📝 Try Sample Texts",
            command=self.show_sample_texts,
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            relief='flat',
            cursor='hand2'
        )
        sample_btn.pack(pady=10)
        
    def create_progress_bars(self):
        """Create progress bars for detailed sentiment scores"""
        # Positive score
        pos_frame = tk.Frame(self.scores_frame, bg='#f0f0f0')
        pos_frame.pack(fill='x', pady=2)
        
        tk.Label(pos_frame, text="Positive:", font=('Arial', 9), bg='#f0f0f0', width=10, anchor='w').pack(side='left')
        self.pos_progress = ttk.Progressbar(pos_frame, length=200, mode='determinate')
        self.pos_progress.pack(side='left', padx=(5, 10))
        self.pos_score_label = tk.Label(pos_frame, text="0.0", font=('Arial', 9), bg='#f0f0f0', width=5)
        self.pos_score_label.pack(side='left')
        
        # Negative score
        neg_frame = tk.Frame(self.scores_frame, bg='#f0f0f0')
        neg_frame.pack(fill='x', pady=2)
        
        tk.Label(neg_frame, text="Negative:", font=('Arial', 9), bg='#f0f0f0', width=10, anchor='w').pack(side='left')
        self.neg_progress = ttk.Progressbar(neg_frame, length=200, mode='determinate')
        self.neg_progress.pack(side='left', padx=(5, 10))
        self.neg_score_label = tk.Label(neg_frame, text="0.0", font=('Arial', 9), bg='#f0f0f0', width=5)
        self.neg_score_label.pack(side='left')
        
        # Neutral score
        neu_frame = tk.Frame(self.scores_frame, bg='#f0f0f0')
        neu_frame.pack(fill='x', pady=2)
        
        tk.Label(neu_frame, text="Neutral:", font=('Arial', 9), bg='#f0f0f0', width=10, anchor='w').pack(side='left')
        self.neu_progress = ttk.Progressbar(neu_frame, length=200, mode='determinate')
        self.neu_progress.pack(side='left', padx=(5, 10))
        self.neu_score_label = tk.Label(neu_frame, text="0.0", font=('Arial', 9), bg='#f0f0f0', width=5)
        self.neu_score_label.pack(side='left')
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Basic cleaning while preserving sentiment-carrying elements
        text = re.sub(r'[^\w\s!?.,;:-]', '', text)
        
        return text
        
    def analyze_sentiment(self):
        """Main sentiment analysis function"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Input Required", "Please enter some text to analyze!")
            return
            
        # Show loading state
        self.analyze_btn.config(text="Analyzing...", state='disabled')
        self.sentiment_result.config(text="Processing your text...", fg='#f39c12')
        
        # Run analysis in separate thread to keep UI responsive
        threading.Thread(target=self.perform_analysis, args=(text,), daemon=True).start()
        
    def perform_analysis(self, text):
        """Perform the actual sentiment analysis"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Get sentiment scores using VADER
            scores = self.sia.polarity_scores(processed_text)
            
            # Update UI in main thread
            self.root.after(0, self.update_results, scores, processed_text)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
            
    def update_results(self, scores, text):
        """Update UI with analysis results"""
        # Determine overall sentiment
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "POSITIVE 😊"
            color = '#27ae60'
            emoji = "😊"
        elif compound <= -0.05:
            sentiment = "NEGATIVE 😔"
            color = '#e74c3c'
            emoji = "😔"
        else:
            sentiment = "NEUTRAL 😐"
            color = '#f39c12'
            emoji = "😐"
            
        # Update main result
        result_text = f"{emoji} {sentiment}\nConfidence: {abs(compound):.3f}"
        self.sentiment_result.config(text=result_text, fg=color, bg='white')
        
        # Update progress bars
        self.pos_progress['value'] = scores['pos'] * 100
        self.pos_score_label.config(text=f"{scores['pos']:.3f}")
        
        self.neg_progress['value'] = scores['neg'] * 100
        self.neg_score_label.config(text=f"{scores['neg']:.3f}")
        
        self.neu_progress['value'] = scores['neu'] * 100
        self.neu_score_label.config(text=f"{scores['neu']:.3f}")
        
        # Reset button
        self.analyze_btn.config(text="🔍 Analyze Sentiment", state='normal')
        
    def show_error(self, error_msg):
        """Handle and display errors"""
        messagebox.showerror("Analysis Error", f"An error occurred: {error_msg}")
        self.sentiment_result.config(text="Error in analysis. Please try again.", fg='#e74c3c')
        self.analyze_btn.config(text="🔍 Analyze Sentiment", state='normal')
        
    def show_sample_texts(self):
        """Show sample texts for testing"""
        samples = [
            "I love this amazing product! It works perfectly!",
            "This is the worst experience I've ever had.",
            "The weather is okay today.",
            "I'm so excited about this new opportunity!",
            "I hate waiting in long lines.",
            "The book was interesting but not exceptional."
        ]
        
        # Create sample selection window
        sample_window = tk.Toplevel(self.root)
        sample_window.title("Sample Texts")
        sample_window.geometry("400x300")
        sample_window.configure(bg='#f0f0f0')
        
        tk.Label(
            sample_window,
            text="Click on any sample text to try it:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0'
        ).pack(pady=10)
        
        for i, sample in enumerate(samples):
            btn = tk.Button(
                sample_window,
                text=sample,
                wraplength=350,
                command=lambda s=sample: self.use_sample_text(s, sample_window),
                font=('Arial', 10),
                bg='white',
                relief='solid',
                borderwidth=1,
                padx=10,
                pady=5,
                cursor='hand2'
            )
            btn.pack(pady=5, padx=10, fill='x')
            
    def use_sample_text(self, sample_text, window):
        """Use selected sample text"""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", sample_text)
        window.destroy()
        self.analyze_sentiment()

def main():
    """Main function to run the app"""
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()