import xarray, netCDF4
from esgcet.scan.handler_base import ESGPubHandlerBase
import os.path
import numpy as np

class ESGPubXArrayHandler(ESGPubHandlerBase):

    @staticmethod
    def xarray_load(map_data):
        datafile = map_data[0][1]
        destpath = os.path.dirname(datafile)

        filespec = f"{destpath}/*.nc"

        time_coder = xarray.coders.CFDatetimeCoder(use_cftime=True)
        res = xarray.open_mfdataset(
            filespec,
            decode_times=time_coder,
            data_vars='all'
        )
        return res

    def get_attrs_dict(self, scanobj):
        return scanobj.attrs

    def get_scanfile_dict(self, scandata, mapdata):
        ret = {}
        for rec in mapdata:
            fn = rec['file']
            ds = netCDF4.Dataset(fn)
            try:
                ret[fn] = {"tracking_id": ds.tracking_id}
            except:
                self.publog.warn("Tracking ID not found")
                ret[fn] = {}
        return ret
    
    def get_variables(self, scanobj):
        res = {}
        for x in scanobj.variables:
            res[x] = scanobj.variables[x].attrs
        return res
    
    def get_variable_list(self, variable):
        return [x for x in variable]

    def _get_time_str(self, timeval):
        if hasattr(timeval, "item"):
            timeval = timeval.item()
        if type(timeval) is float or type(timeval) is int:
            x = str(timeval)
            idx = x.index('.')
            return x[:idx] + 'Z'
        else:
            return timeval.isoformat(timespec="seconds") + "Z"


    def _get_item(self, obj):
        if hasattr(obj, "compute"):
            obj = obj.compute()
        return obj.item()


    def _get_min_max_bounds(self, scanobj, var):

        if var.name not in scanobj.coords:
            raise ValueError("_get_min_max_bounds called on "
                             f"non-coordinate variable {var.name}")
        
        stdname = var.attrs.get("standard_name")

        # use the bounds variable instead if available, so that the range
        # that is returned will include the bounds and not just the central value
        bounds_var = var.attrs.get("bounds")
        if bounds_var is not None:
            var = scanobj[bounds_var]
            using_bounds_var = True
        else:
            using_bounds_var = False

        # undo any broadcasting in time that xarray may have done
        # for non-time variable (seems to do this for vertices array
        # of original shape (ny, nx, 4) when opening multiple files)
        if stdname != "time":
            dims = var.dims
            if dims and dims[0] == "time":
                var = var[0]

        shape = var.shape
        if not using_bounds_var and len(shape) == 1:
            # 1d coordinate variable
            minmax = self._get_item(var[0]), self._get_item(var[-1])
        elif using_bounds_var and len(shape) == 2 and shape[1] == 2:
            # bounds variable of expected shape for 1d coordinate variable
            minmax = self._get_item(var[0][0]), self._get_item(var[-1][1])
        else:
            # complex grid (>1d coordinate variable)
            minmax = (var.values.min(), var.values.max())

        # ensure that max >= min in most cases (even if e.g. data has lats
        # from north to south), but NOT for longitude because it is valid to
        # have min > max numerically due to wrapping, so don't disrupt this
        # because swapping the ordering changes the meaning in non-global case

        if stdname != "longitude" and minmax[1] < minmax[0]:
            minmax = (minmax[1], minmax[0])

        return minmax


    def _get_coord_var_by_stdname(self, scanobj, stdname):
        for coord in scanobj.coords:
            var = scanobj[coord]
            if var.attrs.get("standard_name") == stdname:
                return var
        return None


    def set_bounds(self, record, scanobj):

        geo_units = []

        for (stdname, bounds_names, conv) in [
                ("latitude", ("south_degrees", "north_degrees"), None),
                ("longitude", ("west_degrees", "east_degrees"), None),
                ("time", ("datetime_start", "datetime_end"), self._get_time_str),
                ("air_pressure", ("height_top", "height_bottom"), None),
        ]:
            var = self._get_coord_var_by_stdname(scanobj, stdname)
            if var is not None:
                if len(var.shape) > 0 and var.size > 0:
                    minmax = self._get_min_max_bounds(scanobj, var)
                    if conv is not None:
                        minmax = (conv(minmax[0]), conv(minmax[1]))
                    record[bounds_names[0]], record[bounds_names[1]] = minmax
                    if "units" in var.attrs:
                        geo_units.append(var.units)
                else:
                    self.publog.warn(f"{stdname} found but len 0")


        if len(geo_units) > 0:
            record["geo_units"] = geo_units
