# Use the official NVIDIA JAX Release 25.10 container (contains CUDA 13 & JAX 0.7.2)
FROM nvcr.io/nvidia/jax:25.10-py3

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 1. Setup uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/usr

WORKDIR /app

# 2. Lock the NVIDIA-provided versions so uv knows about them
#    We capture exactly what is installed (jax==0.7.2, flax, etc.)
RUN pip freeze | grep -E "^(jax|jaxlib|flax|optax|orbax)" > /tmp/constraints.txt

# 3. Modify pyproject.toml to REMOVE the "jax[cuda12]" requirement.
#    This prevents uv from trying to "downgrade" to CUDA 12 components.
COPY pyproject.toml .
RUN sed -i '/"jax/d' pyproject.toml

# 4. Install other dependencies
#    - We rely on constraints.txt to force using the system JAX
#    - We install using --system to merge with the NVIDIA environment
RUN uv pip install --system --break-system-packages --no-cache -c /tmp/constraints.txt -e .

COPY src/ src/
COPY recall2imagine/ recall2imagine/
COPY scripts/ scripts/
COPY README.md .
COPY scripts/ scripts/
COPY assets/ assets/
#COPY .env .

# 5. EXECUTION
#    CRITICAL: Use 'python', NOT 'uv run'.
#    'uv run' re-evaluates dependencies at runtime and will try to reinstall JAX.
RUN chmod +x scripts/paper_experiments/run_default_paper_exp_local.sh
ENTRYPOINT ["scripts/paper_experiments/run_default_paper_exp_local.sh"]