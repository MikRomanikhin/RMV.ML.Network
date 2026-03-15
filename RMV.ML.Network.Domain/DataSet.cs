
namespace RMV.ML.Network.Domain;

/// <summary>
/// Represents a collection of paired input and output data sets for machine learning or statistical analysis.
/// </summary>
/// <remarks>
/// The DataSet class is used to store feature vectors and corresponding target values for supervised learning tasks. 
/// Each element in the Source list should correspond to the element at the same index in the Target list. 
/// </remarks>
public class DataSet( List<double[]> source, List<double[]> target )
{
	/// <summary>
	/// Collection of source data arrays.
	/// </summary>
	public List<double[]> Source { get; set; } = source;

	/// <summary>
	/// Collection of target data arrays.
	/// </summary>
	public List<double[]> Target { get; set; } = target;

	/// <summary>
	/// Returns the index of the maximum value within the item array at the specified position.
	/// </summary>
	/// <param name="i">The zero-based index of the item array in the collection to search.</param>
	/// <returns>The zero-based index of the maximum value within the specified item array, or -1 
	/// if the array is empty or the maximum value is not found.</returns>
	public int GetMaxItemIndex( int i ) => Array.IndexOf( this.Target[ i ], this.Target[ i ].Max() );
}
