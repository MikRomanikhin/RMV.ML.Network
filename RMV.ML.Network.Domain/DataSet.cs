
using RMV.ML.Network.Common;

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
	/// Builds random batch of source data arrays from the collection.
	/// </summary>
	/// <param name="batchSize">The number of items to include in the batch.</param>	
	public DataSet GetRandomBatch( int batchSize )
	{
		var source = new List<double[]>( batchSize );
		var target = new List<double[]>( batchSize );

		var indexes = Random.Shared.GetItems( Enumerable.Range( 0, this.Source.Count ).ToArray(), batchSize );

		foreach( int index in indexes )
		{
			source.Add( this.Source[ index ] );
			target.Add( this.Target[ index ] );
		}

		return new DataSet( source, target );
	}
	

	/// <summary>
	/// Determines whether the maximum item index for the specified value equals the provided index.
	/// </summary>
	/// <param name="i">The value for which to retrieve the maximum item index.</param>
	/// <param name="index">The index to compare against the maximum item index of the specified value.</param>	
	public bool Match( int i, int index ) => this.GetMaxItemIndex( i ) == index;


	/// <summary>
	/// Calculates index of the maximum value within the item array at the specified position.
	/// </summary>
	/// <param name="i">The zero-based index of the item array in the collection to search.</param>
	/// <returns>The zero-based index of the maximum value within the specified item array, or -1 
	/// if the array is empty or the maximum value is not found.</returns>
	int GetMaxItemIndex( int i ) => Array.IndexOf( this.Target[ i ], this.Target[ i ].Max() );

}
