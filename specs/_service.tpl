<services>
	<service name="tar_scm">
		<param name="scm">git</param>
		<param name="url">https://github.com/CESNET/GPUJPEG.git</param>
		<param name="version">20170209</param>
		<param name="revision">master</param>
		<param name="filename">libgpujpeg</param>
		<param name="package-meta">yes</param>
		<param name="submodules">enable</param>
	</service>
	<service name="extract_file">
		<param name="archive">*libgpujpeg*.tar</param>
		<param name="files">*/specs/*</param>
	</service>
	<service name="recompress">
		<param name="file">*libgpujpeg*.tar</param>
		<param name="compression">bz2</param>
	</service>
</services>
