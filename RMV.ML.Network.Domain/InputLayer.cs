using System.Text;

using RMV.ML.Network.Configuration;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Input layer of a neural network, responsible for receiving and holding the initial input values
/// </summary>
/// <remarks>
/// This layer serves as the entry point for data into the neural network. It does not perform any computation 
/// or transformation on the input values, but simply stores them for use by subsequent layers.
/// </remarks>
/// <param name="nodes">The number of input nodes in the layer. Must be greater than zero.</param>
/// <param name="settings">The application settings used to configure the layer's behavior.</param>
class InputLayer( int nodes, AppSettings settings ) : BaseLayer( nodes, settings )
{
	/// <summary>	
	/// Feed forward  with the specified input pattern.
	/// </summary>
	/// <param name="input">The input pattern</param>
	public override void Forward( double[]? input )
	{
		ArgumentNullException.ThrowIfNull( input );

#if DEBUG
		for( int i = 0; i < this.Nodes.Count; i++ ) this.Nodes[ i ].Value = input[ i ];
#else
		Parallel.For( 0, this.Nodes.Count, i => this.Nodes[ i ].Value = input[ i ] );
#endif
	}

	/// <summary>
	/// Throws an exception to indicate that backpropagation is not supported for this layer
	/// </summary>	
	public override void Backward( double[]? output = null )
	{
		throw new InvalidOperationException( "Input layer does not support backpropagation." );
	}

	/// <summary>
	/// String representation of the layer, including nodes
	/// </summary>	
	public override string ToString()
	{
		int count = 0;
		var sb = new StringBuilder();

		sb.AppendLine().Append( "Layer " ).Append( count ).Append( " Type: Input" );		

		sb.AppendLine();

		this.Nodes.ForEach( n => sb.AppendLine( n.ToString() ) );

		return sb.ToString();
	}
}
