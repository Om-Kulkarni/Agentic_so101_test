# Agentic_so101_test
Testing agentic capabilities of so101 by making it perform easy pick and place tasks various colored cubes

# Agentic_so101_test
Testing agentic capabilities of so101 by making it perform easy pick and place tasks various colored cubes

## Environment Setup

This project uses Conda for environment management and Poetry for dependency management.

### Prerequisites

- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Om-Kulkarni/Agentic_so101_test
   cd Agentic_so101_test
   ```

2. **Create and activate the Conda environment**
   ```bash
   conda create -n agentic_so101 python=3.10
   conda activate agentic_so101
   ```

3. **Install Poetry**
   ```bash
   conda install -c conda-forge poetry
   ```

4. **Install project dependencies using Poetry**
   ```bash
   poetry install
   ```

5. **Verify the installation**
   ```bash
   poetry run python -c "import lerobot; print(lerobot.__version__)"
   poetry run python -c "import google.generativeai; print('google-genai installed')"
   ```

### Development

- Always activate the environment before working on the project:
  ```bash
  conda activate agentic_so101
  ```

- To add new dependencies:
  ```bash
  poetry add package-name
  ```

- To run your scripts:
  ```bash
  poetry run python your_script.py
  ```

### Notes

- This project uses lerobot from Hugging Face's GitHub repository with pi0 and feetech extras
- Google Generative AI (google-genai) is used for the agentic capabilities