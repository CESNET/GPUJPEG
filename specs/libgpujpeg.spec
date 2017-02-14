Name:	libgpujpeg
Version:	20170209
Release:	00%{?dist}
Summary:	Experimental GPU JPEG codec implementation
Group:	Development/Libraries

License:	BSD-2-Clause
URL:		https://github.com/CESNET/GPUJPEG
Source0:	libgpujpeg-%{version}.tar.bz2
Source1:	libgpujpeg-rpmlintrc

BuildRequires:	gcc-c++,make,automake,autoconf,glew-devel,libtool
#BuildRequires:	cuda-drivers = 340.29-0.x86_64,akmod-nvidia
# see https://en.opensuse.org/openSUSE:Build_Service_cross_distribution_howto
%if 0%{?fedora} > 1 && 0%{?fedora} < 21
BuildRequires:	cuda-core-6-5,cuda-command-line-tools-6-5,cuda-cudart-dev-6-5
%else
BuildRequires:	cuda-core-8-0,cuda-command-line-tools-8-0,cuda-cudart-dev-8-0
BuildRequires:	clang
%endif

%description
GPUJPEG is compression and decompression library accelerated on GPU. GPU
acceleration allows for compression and decompression of full HD images
up to 750fps, 4K images up to 125fps, 8K images up to 33fps on one card.
GPUJPEG was developed as a part of UltraGrid (http://ultragrid.sitola.cz ).

Acknowledgments: project Large Infrastructure CESNET (LM2010005)
-- http://www.cesnet.cz .

Reference paper:
HOLUB, Petr - ŠROM, Martin - PULEC, Martin - MATELA, Jiří - JIRMAN, Martin.
GPU-accelerated DXT and JPEG compression schemes for low-latency network
transmissions of HD, 2K, and 4K video. Future Generation Computer Systems,
Amsterdam, The Netherlands, Elsevier Science, Nizozemsko. ISSN 0167-739X,
2013, vol. 29, no. 8, 1991–2006-16


%define _use_internal_dependency_generator 0
%define __find_requires	/bin/bash -c "/usr/lib/rpm/find-requires | sed -e '\
/^libcudart\.so/d; \
/^libnpp\.so/d; \
'"
%define __find_provides	/bin/bash -c "/usr/lib/rpm/find-provides"

%package	devel
Requires:	libgpujpeg = %{version}-%{release}
Summary:	Development files for libgpujpeg
%description	devel
Libgpujpeg headers plus pkg-config control files.

%package	tools
Requires:	libgpujpeg = %{version}-%{release}
Summary:	Wrapper for libgpujpeg
%description	tools
Simple wrapper binary for libgpujpeg.

%prep
%setup -q

%build
./autogen.sh || true
%configure --docdir=%{_docdir} --disable-static --enable-opengl --with-cuda-host-compiler=clang --with-cuda=/usr/local/cuda-8.0
make %{?_smp_mflags}

%install
#export QA_RPATHS=0x0001
rm -rf $RPM_BUILD_ROOT
%makeinstall

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%defattr(-,root,root,-)
%doc AUTHORS ChangeLog COPYING NEWS README
%{_libdir}/libgpujpeg*.so.*

%files tools
%defattr(-,root,root,-)
%{_bindir}/gpujpeg

%files	devel
%defattr(-,root,root,-)
%exclude %{_libdir}/%{name}.la
%{_includedir}/libgpujpeg
%{_libdir}/libgpujpeg.so
%{_libdir}/pkgconfig/%{name}.pc
%{_libdir}/libgpujpeg
%{_libdir}/libgpujpeg/config.h

%changelog
* Tue Feb 14 2017 Lukas Rucka <xrucka@fi.muni.cz> 20170209
- integrated build patches into upstream, merged package specifications

* Wed Mar 4 2015 Lukas Rucka <xrucka@fi.muni.cz>
- upstream commit caa5ece51bc11fd52d8625f0975a07620102fb30

* Wed Oct 23 2013 Lukas Rucka <xrucka@fi.muni.cz>
- the gpujpeg library is now considered stable enough to claim 1.0.0 version
- pure upstream

* Mon Jul 02 2012 Lukas Rucka <xrucka@fi.muni.cz>
- first packaged version of the library
- switched to autotools build subsystem

