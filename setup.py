import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="acdc_py",
    version="1.0.0",
    author="Alexander L.E. Wang",
    author_email="aw3436@cumc.columbia.edu",
    packages=["acdc_py"],
    description="A package to quickly identify unbiased graph-based clusterings via parameter optimization in Python",
    long_description="This Scanpy-compatible package provides the ability to iterative compute clusterings using graph-based on combinations of resolution and KNN. At each point in this search space, a metric, such as silhouette score, is computed. The solution from this search space that maximizes the metric results in an optimal unbiased clustering solution. This process is sped up by our subsampling and diffusion feature that provides a near-identical solution in a much shorter time.",
    long_description_content_type="text/markdown",
    url="https://github.com/alexanderlewis99/acdc_py",
    license='MIT',
    python_requires='>=3.8',
    install_requires=[
        "scipy",
        "tqdm",
        "scanpy",
        "anndata",
        "pandas",
        "numpy",
        "joblib",
        "leidenalg",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "datetime"
    ]
)
