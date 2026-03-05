namespace MLNet.ImageGeneration.OnnxGenAI;

/// <summary>
/// Simple Euler Discrete scheduler for the Stable Diffusion denoising loop.
/// Manages the noise schedule and computes each denoising step.
/// For SD-Turbo, 1-4 inference steps suffice.
/// </summary>
internal sealed class EulerDiscreteScheduler
{
    private readonly int _numTrainTimesteps;
    private readonly float _betaStart;
    private readonly float _betaEnd;
    private float[] _alphasCumulativeProduct = [];
    private float[] _sigmas = [];

    public float[] Timesteps { get; private set; } = [];
    public float InitNoiseSigma { get; private set; }

    public EulerDiscreteScheduler(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f)
    {
        _numTrainTimesteps = numTrainTimesteps;
        _betaStart = betaStart;
        _betaEnd = betaEnd;
        ComputeAlphas();
    }

    private void ComputeAlphas()
    {
        var betas = new float[_numTrainTimesteps];
        float sqrtStart = MathF.Sqrt(_betaStart);
        float sqrtEnd = MathF.Sqrt(_betaEnd);
        for (int i = 0; i < _numTrainTimesteps; i++)
        {
            float t = (float)i / (_numTrainTimesteps - 1);
            float beta = sqrtStart + t * (sqrtEnd - sqrtStart);
            betas[i] = beta * beta;
        }

        _alphasCumulativeProduct = new float[_numTrainTimesteps];
        float cumprod = 1.0f;
        for (int i = 0; i < _numTrainTimesteps; i++)
        {
            cumprod *= (1.0f - betas[i]);
            _alphasCumulativeProduct[i] = cumprod;
        }
    }

    public void SetTimesteps(int numInferenceSteps)
    {
        var stepRatio = (float)_numTrainTimesteps / numInferenceSteps;
        Timesteps = new float[numInferenceSteps];
        _sigmas = new float[numInferenceSteps + 1];

        for (int i = 0; i < numInferenceSteps; i++)
        {
            int t = (int)((_numTrainTimesteps - 1) - i * stepRatio);
            Timesteps[i] = t;
            _sigmas[i] = MathF.Sqrt((1 - _alphasCumulativeProduct[t]) / _alphasCumulativeProduct[t]);
        }
        _sigmas[numInferenceSteps] = 0;
        InitNoiseSigma = _sigmas[0];
    }

    public float[] Step(float[] modelOutput, int stepIndex, float[] sample)
    {
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];

        // Euler step: x = x + (sigma_next - sigma) * model_output
        var result = new float[sample.Length];
        float dt = sigmaNext - sigma;
        for (int i = 0; i < sample.Length; i++)
        {
            result[i] = sample[i] + dt * modelOutput[i];
        }
        return result;
    }

    public float[] ScaleModelInput(float[] sample, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float scale = MathF.Sqrt(sigma * sigma + 1);
        var result = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
            result[i] = sample[i] / scale;
        return result;
    }
}
