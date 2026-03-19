
using Microsoft.Extensions.Configuration;

namespace RMV.ML.Network.Common;

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
	/// Error threshold
	/// </summary>
	//public double Eps { get; set; }

	/// <summary>
	/// Learning rate
	/// </summary>
	[ConfigurationKeyName( "learning" )]
	public double Rate { get; set; }

	/// <summary>
	/// Momentum parameter
	/// </summary>
	public double Momentum { get; set; }

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
	public int Validate { get; set; }

	/// <summary>
	/// Train data file path
	/// </summary>
	public required string Train { get; set; }

	/// <summary>
	/// Test data file path
	/// </summary>
	public required string Test { get; set; }
	
	/// <summary>
	/// Errors file path
	/// </summary>
	public string Errors { get; set; }		
}
