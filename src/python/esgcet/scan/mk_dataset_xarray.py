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


    def _undo_time_broadcast(self, var):
        dims = var.dims
        if dims and dims[0] == "time":
            return var[0]
        else:
            return var


    def _get_min_max_bounds(self, scanobj, var):

        if var.name not in scanobj.coords:
            raise ValueError("_get_min_max_bounds called on "
                             f"non-coordinate variable {var.name}")
        
        stdname = var.attrs.get("standard_name")

        # use the bounds variable instead if available, so that the range
        # that is returned will include the bounds and not just the central value
        bounds_var_name = var.attrs.get("bounds")
        if bounds_var_name is not None:
            bounds_var = scanobj[bounds_var_name]
            using_bounds_var = True
            self.publog.info(f"{stdname} has bounds var")
        else:
            using_bounds_var = False
            self.publog.info(f"{stdname} no bounds var")

        # undo any broadcasting in time that xarray may have done
        # for non-time variable (seems to do this for vertices array
        # of original shape (ny, nx, 4) when opening multiple files)
        if stdname != "time":
            var = self._undo_time_broadcast(var)
            if using_bounds_var:
                bounds_var = self._undo_time_broadcast(bounds_var)

        shape = var.shape
        if not using_bounds_var and len(shape) == 1:

            # 1d coordinate variable
            first = self._get_item(var[0])
            last = self._get_item(var[-1])
            minmax = (first, last)

            # deal with a special case
            # no bounds variable specified, but values on a regular
            # longitude grid make it obvious that this is really global
            if (stdname == "longitude"
                and var.size > 1):

                interval = self._get_item(var[1]) - first
                after_last = last + interval
                if abs(after_last - first - 360) < 1e-3:
                    minmax = (first, after_last)

        elif using_bounds_var and len(shape) == 2 and shape[1] == 2:
            # bounds variable of expected shape for 1d coordinate variable
            minmax = (self._get_item(bounds_var[0][0]),
                      self._get_item(bounds_var[-1][1]))

        else:
            # complex grid (>1d coordinate variable)
            # use the extreme values found

            # Where a bounds var is used, look in both the var and the bounds var.
            # Normally the extreme values would be found in the bounds var, but in one
            # example (CNRM-CM6-1-HR), for longitude, a gridbox *centre* was on the
            # 180 meridian, hence why looking also in the main var.
            if using_bounds_var:
                minmax = (min(var.values.min(), bounds_var.values.min()),
                          max(var.values.max(), bounds_var.values.max()))
            else:
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
