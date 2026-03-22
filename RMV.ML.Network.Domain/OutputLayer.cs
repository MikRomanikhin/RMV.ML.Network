using System.Text;

using RMV.ML.Network.Configuration;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Output layer is responsible for producing the final output values and computing the overall error during training.
/// </summary>
/// <remarks>
/// The output layer applies the softmax function to its node values during the forward pass, ensuring the outputs 
/// form a valid probability distribution. During backpropagation, it calculates the error based on the provided 
/// target output pattern. This layer is typically used as the final layer in classification neural networks.</remarks>
/// <param name="nodes">The number of nodes (neurons) in the output layer. Must be greater than zero.</param>
/// <param name="settings">The application settings that configure the behavior of the layer, such as learning rate or activation functions.</param>
/// <param name="source">The preceding layer that provides input to this output layer. Cannot be null.</param>
class OutputLayer( int nodes, AppSettings settings, BaseLayer source ) : BaseLayer( nodes, settings, source )
{
	/// <summary>
	/// Output Layer Error
	/// </summary>
	public double Error { get; private set; }

	/// <summary>
	/// Feed forward
	/// </summary>
	public override void Forward( double[]? input = null ) //=> this.Nodes.ForEach( n => n.Forward() );
	{
#if DEBUG
		this.Nodes.ForEach( n => n.Forward() );
#else
		Parallel.ForEach( this.Nodes, n => n.Forward() );
#endif
		Softmax();
	}

	/// <summary>
	/// Applies the softmax function to the output layer nodes	
	/// </summary>
	void Softmax()
	{
		double max = this.Nodes.Max( n => n.RawValue );

		this.Nodes.ForEach( n => n.Value = Math.Exp( n.RawValue - max ) );

		double sum = this.Nodes.Sum( n => n.Value );

		this.Nodes.ForEach( n => n.Value /= sum );
	}


	/// <summary>
	/// Output layer learning with the specified output pattern and error calculation.
	/// </summary>
	/// <param name="output">The output pattern</param>
	public override void Backward( double[]? output )
	{
		if (output == null) throw new ArgumentNullException(nameof(output));

		this.Error = this.Nodes.Select( ( n, i ) => n.Backward( output[ i ] ) ).Sum();
	}

	/// <summary>
	/// Calculates the index of the output node with the highest value
	/// </summary>	
	internal int GetMaxItemIndex()
	{
		int maxIndex = 0;
		double maxValue = this.Nodes[ 0 ].Value;

		for( int i = 1; i < this.Nodes.Count; i++ )
		{
			double value = this.Nodes[ i ].Value;

			if( value > maxValue )
			{
				maxValue = value;
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	/// <summary>
	/// String representation of the layer, including nodes
	/// </summary>	
	public override string ToString()
	{
		int count = 0;
		var sb = new StringBuilder();

		sb.AppendLine().Append( "Layer " ).Append( count ).Append( " Type: Output" );

		sb.Append( " Source: " ).Append( this.Source!.GetType().Name );
		sb.Append( " Target: " ).Append( this.Target!.GetType().Name );

		sb.AppendLine();

		this.Nodes.ForEach( n => sb.AppendLine( n.ToString() ) );

		return sb.ToString();
	}
}
