using System.Text;

using RMV.ML.Network.Common;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Hidden layer responsible for processing inputs and producing intermediate outputs using an activation function.
/// </summary>
/// <remarks>
/// This layer typically applies a non-linear transformation to its inputs, enabling the network to learn complex patterns. 
/// This class uses the ReLU activation function by default and supports both forward and backward propagation.  
/// </remarks>
class HiddenLayer : BaseLayer
{
	/// <summary>
	/// Initializes a new instance with the specified number of nodes, application settings, and source layer.
	/// </summary>
	/// <param name="nodes">The number of nodes in the hidden layer. Must be a positive integer.</param>
	/// <param name="settings">The application settings to configure the layer's behavior. Cannot be null.</param>
	/// <param name="source">The preceding layer from which this hidden layer receives input. Cannot be null.</param>
	public HiddenLayer( int nodes, AppSettings settings, BaseLayer source ) : base( nodes, settings, source )
	{
		this.activation = new Relu();
	}

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
		
	}

	/// <summary>
	/// Backward pass, propagating gradients for learning.
	/// </summary>		
	public override void Backward( double[]? output = null ) //=> this.Nodes.ForEach( n => n.Back() );
	{
#if DEBUG
		this.Nodes.ForEach( n => n.Backward() );
#else
		Parallel.ForEach( this.Nodes, n => n.Backward() );
#endif
	}

	/// <summary>
	/// String representation of the layer, including nodes
	/// </summary>	
	public override string ToString()
	{
		int count = 0;
		var sb = new StringBuilder();

		sb.AppendLine().Append( "Layer " ).Append( count ).Append( " Type: Hidden" );

		sb.Append( " Source: " ).Append( this.Source!.GetType().Name );
		sb.Append( " Target: " ).Append( this.Target!.GetType().Name );

		sb.AppendLine();

		this.Nodes.ForEach( n => sb.AppendLine( n.ToString() ) );

		return sb.ToString();
	}
}
