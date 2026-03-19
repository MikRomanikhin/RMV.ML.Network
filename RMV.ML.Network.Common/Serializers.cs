using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Xml;
using System.Xml.Serialization;

namespace RMV.ML.Network.Common;

/// <summary>
/// XML and JSON Serialization extenders
/// </summary>
public static class Serialization
{
	#region Xml -------------------------------------------------------------   

	/// <summary>
	/// Serialize to XML with namespace
	/// </summary>
	/// <typeparam name="T">target object type</typeparam>
	/// <param name="target">target object</param>
	/// <param name="XmlSerializerNamespaces">namespace</param>		
	/// <returns>XML string</returns>
	public static string? ToXml<T>( this T target, XmlSerializerNamespaces? namespaces = null )
	{
		if( target == null ) return null;

		var stringWriter = new StringWriter();

		using var xmlWriter = XmlWriter.Create( stringWriter, new XmlWriterSettings { OmitXmlDeclaration = true } );

		var serializer = new XmlSerializer( typeof( T ) );

		var blank = new XmlSerializerNamespaces();
		blank.Add( "", "" );

		serializer.Serialize( xmlWriter, target, namespaces ?? blank );

		return stringWriter.ToString();
	}

	/// <summary>
	/// Reconstruct an object from an XML string
	/// </summary>
	/// <typeparam name="T">object type</typeparam>
	/// <param name="source">source string</param>		
	/// <returns>deserialized object</returns>
	public static T? FromXml<T>( this string source ) where T : class
	{
		if( source.IsNullOrEmpty() ) return null;

		var serializer = new XmlSerializer( typeof( T ) );

		using var stream = new MemoryStream( new UTF8Encoding().GetBytes( source ) );

		return ( T? )serializer.Deserialize( stream );
	}

	#endregion	


	#region Json ------------------------------------------------------------

	/// <summary>
	/// Json serializing
	/// </summary>
	/// <typeparam name="T">target object type</typeparam>
	/// <param name="target">target object</param>
	/// <returns>Json string</returns>
	public static string? ToJson<T>( this T? target, JsonSerializerOptions? options = null )
	{
		return target is null ? default : JsonSerializer.Serialize( target, options ?? DefaultJsonSerializerOptions );
	}

	/// <summary>
	/// Json serializing async
	/// </summary>
	/// <typeparam name="T">target object type</typeparam>
	/// <param name="target">target object</param>
	/// <returns>Json string</returns>
	public static async Task<string?> ToJsonAsync<T>( this T? target, JsonSerializerOptions? options = null )
	{
		if( target == null ) return null;

		using var stream = new MemoryStream();
		await JsonSerializer.SerializeAsync( stream, target, options ?? DefaultJsonSerializerOptions );

		stream.Position = 0;
		using var reader = new StreamReader( stream );
		return await reader.ReadToEndAsync();
	}

	/// <summary>
	/// Reconstruct object from Json string
	/// </summary>
	/// <typeparam name="T">target object type</typeparam>
	/// <param name="source">json string</param>
	/// <returns>deserialized object</returns>
	public static T? FromJson<T>( this string? source, JsonSerializerOptions? options = null )
	{		
		return source is null ? default : JsonSerializer.Deserialize<T>( source, options ?? DefaultJsonSerializerOptions );
	}

	/// <summary>
	/// Reconstruct object from Json string async
	/// </summary>
	/// <typeparam name="T">target object type</typeparam>
	/// <param name="source">json string</param>
	/// <returns>deserialized object</returns>
	public static async Task<T?> FromJsonAsync<T>( this string source, JsonSerializerOptions? options = null )
	{
		Stream stream = new MemoryStream( Encoding.UTF8.GetBytes( source ) );

		return await JsonSerializer.DeserializeAsync<T>( stream, options );
	}

	/// <summary>
	/// Creates a deep copy of the specified object by serializing and deserializing it using JSON.
	/// </summary>
	/// <remarks>
	/// JSON serialization may not preserve object references or types not supported by System.Text.Json. 
	/// Use caution with types that have non-serializable members or require custom converters.
	/// </remarks>
	/// <typeparam name="T">The type of the object to copy. Must be serializable by System.Text.Json.</typeparam>
	/// <param name="source">The object to copy</param>
	/// <param name="options">The options to use for JSON serialization and deserialization. If null, default options are used.</param>
	/// <returns>A deep copy of the source object, or null if source is null.</returns>
	public static T? DeepCopy<T>( this T? source, JsonSerializerOptions? options = null )
	{
		return source is null ? default : source.ToJson( options ).FromJson<T?>( options );		
	}

	/// <summary>
	/// Is this a valid JSON string
	/// </summary>
	/// <param name="target">target string</param>
	/// <returns>success indicator</returns>	
	public static bool IsValidJson( this string target )
	{
		if( string.IsNullOrWhiteSpace( target ) )	return false;

		try // Try parsing the JSON
		{
			using JsonDocument doc = JsonDocument.Parse( target );
			return true; // Successfully parsed
		}
		catch( JsonException )
		{			
			return false; // Invalid JSON format
		}
		catch( NotSupportedException )
		{			
			return false; // Input contains unsupported JSON features
		}
	}


	static JsonSerializerOptions DefaultJsonSerializerOptions
	{
		get
		{
			var options = new JsonSerializerOptions {
				DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,//IgnoreNullValues = true,
				WriteIndented = true,
				IgnoreReadOnlyProperties = true,
				PropertyNamingPolicy = JsonNamingPolicy.CamelCase
			};

			options.Converters.Add( new JsonStringEnumConverter( JsonNamingPolicy.CamelCase ) );

			return options;
		}
	}

	#endregion

}
