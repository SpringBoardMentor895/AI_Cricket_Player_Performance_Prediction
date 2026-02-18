import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def run_notebook(notebook_path):
    """Run a Jupyter notebook and report any errors"""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print(f"{'='*60}\n")
    
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': './'}})
        
        # Save the executed notebook
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✅ SUCCESS: {notebook_path} executed without errors!")
        print(f"   Output saved to: {output_path}\n")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in {notebook_path}:")
        print(f"   {str(e)}\n")
        return False

if __name__ == "__main__":
    notebooks = [
        "01_EDA.ipynb",
        "02_Feature_engineering.ipynb",
        "03_ModelTraining.ipynb",
        "04_Streamlit_app.ipynb"
    ]
    
    results = {}
    for nb in notebooks:
        results[nb] = run_notebook(nb)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for nb, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {nb}")
