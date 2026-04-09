using Microsoft.Extensions.Configuration;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Network configuration
/// </summary>
public class AppSettings
{
	/// <summary>
	/// Input layer size
	/// </summary>
	public int Input { get; set; }

	/// <summary>
	/// Hidden layers sizes
	/// </summary>
	public int[] Hidden { get; set; } = [];

	/// <summary>
	/// Output layer size
	/// </summary>
	public int Output { get; set; }

	/// <summary>
	/// Activation function type
	/// </summary>
	public ActivationType Activation { get; set; } = ActivationType.Relu;

	/// <summary>
	/// Learning type
	/// </summary>
	public LearningType Learning { get; set; }

	/// <summary>
	/// Learning rate
	/// </summary>	
	public double Rate { get; set; }

	/// <summary>
	/// Momentum parameter
	/// </summary>
	public double Momentum { get; set; }

	/// <summary>
	/// Optimizer type
	/// </summary>
	public Optimizer Optimizer { get; set; } = Optimizer.SGD;

	/// <summary>
	/// Weight decay parameter
	/// </summary>
	public double Decay { get; set; }	

	/// <summary>
	/// Number of iterations
	/// </summary>
	public int Iterations { get; set; }
	
	/// <summary>
	/// Batch size
	/// </summary>
	public int Batch { get; set; }

	/// <summary>
	/// Print interval
	/// </summary>
	public int Print { get; set; }

	/// <summary>
	/// Validation interval
	/// </summary>
	public int Epoch { get; set; }

	/// <summary>
	/// Train data file path
	/// </summary>
	[ConfigurationKeyName( "train-path" )]
	public required string TrainPath { get; set; }

	/// <summary>
	/// Test data file path
	/// </summary>
	[ConfigurationKeyName( "test-path" )]
	public required string TestPath { get; set; }

	/// <summary>
	/// Errors file path
	/// </summary>
	[ConfigurationKeyName( "error-path" )]
	public string? ErrorPath { get; set; }		
}
