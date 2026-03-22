using RMV.ML.Network.Configuration;
using RMV.ML.Network.Domain;

namespace RMV.ML.Network.Test;

/// <summary>
/// Maps data to the Activation Function range
/// </summary>
public class Mapper( IActivation activation )
{
	/// <summary>
	/// Normalizes input to the Activation Function range using Min-Max scaling
	/// </summary>	
	public IEnumerable<double> NormalizeMinMax( IEnumerable<double> data )
	{
		double min = data.Min();
		double max = data.Max();

		var delta = activation.Max - activation.Min;

		return data.Select( d => ( d - min ) * delta / ( max - min ) + activation.Min );
	}

	/// <summary>
	/// Normalizes input using Z-score normalization
	/// </summary>	
	public static IEnumerable<double> NormalizeZscore( IEnumerable<double> data )
	{
		double mean = data.Average();
		double stdev = data.StdDev();

		return data.Select( d => ( d - mean ) / stdev );
	}

	/// <summary>
	/// Normalizes input to [0, 1] range by dividing by a fixed scale (255 for image data).
	/// </summary>	
	public static IEnumerable<double> Normalize( IEnumerable<double> data ) => data.Select( d => d / 255.0 );	
	

	/// <summary>
	/// Denormalizes data for display
	/// </summary>
	/// <param name="data">data</param>
	/// <param name="max">max value</param>
	/// <param name="min">min value</param>
	/// <returns>denormalized data</returns>
	public IEnumerable<double> Denormalize( IEnumerable<double> data, double max, double min )
	{
		var scale = max - min;

		var delta = activation.Max - activation.Min;

		return data.Select( d => ( d - activation.Min ) * scale / delta + min );
	}
}
