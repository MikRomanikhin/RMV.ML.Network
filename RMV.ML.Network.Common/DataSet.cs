using MathNet.Numerics.LinearAlgebra;

namespace RMV.ML.Network.Common;

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

		var indices = Enumerable.Range( 0, this.Source.Count ).ToArray();
		Random.Shared.Shuffle( indices );

		foreach( int index in indices.Take( batchSize ) )
		{
			source.Add( this.Source[ index ] );
			target.Add( this.Target[ index ] );
		}

		return new DataSet( source, target );
	}


	//public (Matrix<double>, Matrix<double>) GetRandomTrain( int batchSize )
	//{		
	//	var indices = Enumerable.Range( 0, this.Source.Count ).ToArray();
	//	Random.Shared.Shuffle( indices );

	//	// Eagerly materialize arrays to prevent deferred execution issues in MathNet
	//	var trainIndices = indices.Take( batchSize ).ToArray();
	//	var testIndices = indices.Skip( batchSize ).Take( batchSize ).ToArray(); // Use Skip to get different rows

	//	var xBatchRows = trainIndices.Select( i => this.Source[ i ] ).ToArray();
	//	var tBatchRows = testIndices.Select( i => this.Target[ i ] ).ToArray(); // NOTE: You are using Source here, not Target.

	//	var xBatch = Matrix<double>.Build.DenseOfRowArrays( xBatchRows );
	//	var tBatch = Matrix<double>.Build.DenseOfRowArrays( tBatchRows );

	//	return (xBatch, tBatch);
	//}

	/// <summary>
	/// Selects a random batch of source data arrays from the collection and returns them as a tuple of matrices.
	/// </summary>	
	public (Matrix<double>, Matrix<double>) GetRandomTrain( int batchSize )
	{		
		int inputSize = this.Source[ 0 ].Length;
		int outputSize = this.Target[ 0 ].Length;

		var indices = Enumerable.Range( 0, this.Source.Count ).ToArray();
		Random.Shared.Shuffle( indices );

		var xBatch = Matrix<double>.Build.Dense( batchSize, inputSize );
		var tBatch = Matrix<double>.Build.Dense( batchSize, outputSize );

		for( int row = 0; row < batchSize; row++ )
		{
			int index = indices[ row ];

			xBatch.SetRow( row, this.Source[ index ] );
			tBatch.SetRow( row, this.Target[ index ] );
		}

		return (xBatch, tBatch);
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
