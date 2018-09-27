/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.ReportData.TEST_RPT_NAME;
import static org.edgexfoundry.test.data.ReportData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceReportDao;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ReportData;
import org.edgexfoundry.test.data.ScheduleData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class DeviceReportDaoTest {

  @Autowired
  private DeviceReportDao dao;

  @Autowired
  private DeviceReportRepository repos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private ScheduleEventRepository schEventRepos;

  @Autowired
  private ScheduleRepository schRepos;

  private String id;

  @Before
  public void createTestData() {
    DeviceService service = ServiceData.newTestInstance();
    serviceRepos.save(service);
    Device device = DeviceData.newTestInstance();
    device.setService(service);
    deviceRepos.save(device);
    Schedule schedule = ScheduleData.newTestInstance();
    schRepos.save(schedule);
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    schEventRepos.save(event);
    DeviceReport report = newTestInstance();
    repos.save(report);
    id = report.getId();
    assertNotNull("new test Device Report has no identifier", id);
    assertNotNull("Dao is null", dao);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    deviceRepos.deleteAll();
    schEventRepos.deleteAll();
    addrRepos.deleteAll();
    schRepos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testGetByIdOrName() {
    DeviceReport report = new DeviceReport(TEST_RPT_NAME, null, null, null);
    assertNotNull("Device report is null on getByIdOrName with valid name", dao.getByIdOrName(report));
    report.setName(null);
    report.setId(id);
    assertNotNull("Device report is null on getByIdOrName with valid id", dao.getByIdOrName(report));
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("No device report should be found with null on getByIdOrName",
        dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrNameWithBadIdentifiers() {
    DeviceReport report = new DeviceReport(null, null, null, null);
    report.setId("badid");
    assertNull("No device report should be found with bad id on getByIdOrName",
        dao.getByIdOrName(report));
    report.setId(null);
    report.setName("badname");
    assertNull("No device report should be found with bad name on getByIdOrName",
        dao.getByIdOrName(report));
  }

  @Test
  public void testGetOwningServices() {
    DeviceReport report = repos.findOne(id);
    DeviceService service = serviceRepos.findByName(ServiceData.TEST_SERVICE_NAME);
    assertEquals("Did not find the correct associated DeviceService ", service.getId(),
        dao.getOwningService(report).getId());
  }

  @Test
  public void testGetOwningServicesHavingNone() {
    Device device = deviceRepos.findByName(DeviceData.TEST_NAME);
    device.setService(null);
    deviceRepos.save(device);
    serviceRepos.deleteAll();
    assertNull("Should be no owning device services", dao.getOwningService(repos.findOne(id)));
  }

  @Test
  public void testGetOwningServicesWithNull() {
    assertNull("Should be no owning device services", dao.getOwningService(null));
  }

  @Test
  public void testRemoveAssociatedReportsForDevice() {
    dao.removeAssociatedReportsForDevice(deviceRepos.findByName(DeviceData.TEST_NAME));
    assertTrue("Device reports still exist after removing those associated to the device",
        repos.findAll().isEmpty());
  }

  @Test
  public void testGetValueDescriptorsForDeviceReportsAssociatedToDevice() {
    List<String> valueDescriptorNames =
        dao.getValueDescriptorsForDeviceReportsAssociatedToDevice(DeviceData.TEST_NAME);
    assertEquals("lsit of VDs from device reports for assocaited devices not what expected",
        Arrays.asList(ReportData.TEST_EXPECTED), valueDescriptorNames);
  }

}
